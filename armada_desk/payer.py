"""Pay a URL over x402 through Armada, as one of the desk's agents.

Calls Armada's MCP tools (``wallet_get_agent_info``, ``wallet_pay_url``) on
``/mcp`` with the agent's API key as a Bearer token. The key is read from an
environment variable named by the caller and is never logged, saved, or put in
an error message.

The server is stateless, so each tool call is one JSON-RPC POST with no
``initialize``; it answers as JSON or as one SSE frame. Tool results are text.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import requests

logger = logging.getLogger(__name__)

MCP_URL = "https://agentic.anchorage.com/mcp"


class PaymentError(RuntimeError):
    """A payment that didn't deliver the resource. The text never holds the key."""


class TransientError(PaymentError):
    """Worth retrying with the same idempotency key."""


@dataclass
class PaidResponse:
    url: str
    idempotency_key: str
    transaction_id: str | None
    status: int
    body: str
    usd: Decimal | None   # what Armada paid, from its transaction list; None if unread
    replayed: bool = False


_BUDGET_RE = (
    re.compile(r"budget_id:\s*(\S+)"),               # "Spend against → budget_id: <id>"
    re.compile(r"\(id=([^,\s)]+),"),                  # "<name> (id=<id>, period=...)"
)


def parse_budget_id(agent_info: str) -> str:
    """The first admissible budget id in ``wallet_get_agent_info``'s text."""
    section = agent_info.split("Budgets", 1)[-1]
    for rx in _BUDGET_RE:
        m = rx.search(section)
        if m:
            return m.group(1)
    raise PaymentError("wallet_get_agent_info listed no budget this agent can spend against")


def parse_completed(text: str) -> tuple[str | None, int, str, bool]:
    """(transaction id, merchant HTTP status, merchant body, replayed) from a completed payment.

    The text is ``Payment completed (...).\\nTransaction ID: ..\\nHTTP ..\\n<settlement>\\n<body>``,
    behind an ``[IDEMPOTENCY CACHE HIT ...]`` note on a replay.
    """
    replayed = text.startswith("[IDEMPOTENCY CACHE HIT")
    if replayed:
        text = text.split("]\n\n", 1)[-1]
    lines = text.split("\n")
    if lines[0].startswith("HTTP "):           # the URL didn't ask for payment
        return None, int(lines[0].split()[1]), "\n".join(lines[2:]), replayed
    if not lines[0].startswith("Payment completed") or len(lines) < 5:
        raise PaymentError(f"unexpected wallet_pay_url result: {lines[0][:120]}")
    txid = lines[1].removeprefix("Transaction ID:").strip()
    status = int(lines[2].split()[1])
    body = "\n".join(lines[4:])
    body = body.split("\n[note: ", 1)[0]
    return txid, status, body, replayed


class ArmadaPayer:
    """One agent's payer. ``pay`` retries transient failures with the same key."""

    def __init__(self, key_env: str, mcp_url: str = MCP_URL, budget_id: str | None = None,
                 retries: int = 2, backoff: float = 2.0, session: requests.Session | None = None):
        self.key_env = key_env
        self.mcp_url = mcp_url
        self._budget_id = budget_id
        self.retries = retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self._ids = itertools.count(1)

    def _key(self) -> str:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise PaymentError(f"{self.key_env} is unset or empty")
        return key

    def _scrub(self, text: str) -> str:
        key = os.environ.get(self.key_env, "")
        return text.replace(key, "***") if key else text

    def call_tool(self, name: str, arguments: dict) -> str:
        """One MCP ``tools/call``; the result's text, or ``PaymentError`` if it is an error."""
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": "tools/call",
                   "params": {"name": name, "arguments": arguments}}
        headers = {"Authorization": f"Bearer {self._key()}",
                   "Accept": "application/json, text/event-stream",
                   "Content-Type": "application/json"}
        try:
            resp = self.session.post(self.mcp_url, json=payload, headers=headers, timeout=120)
        except requests.RequestException as exc:
            raise TransientError(self._scrub(f"{name}: {type(exc).__name__} reaching Armada")) from None
        if resp.status_code >= 500 or resp.status_code == 429:
            raise TransientError(f"{name}: Armada answered HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise PaymentError(f"{name}: Armada answered HTTP {resp.status_code}"
                               + (" (check the agent key)" if resp.status_code in (401, 403) else ""))
        msg = _jsonrpc_message(resp)
        if "error" in msg:
            raise PaymentError(self._scrub(f"{name}: {msg['error'].get('message', msg['error'])}"))
        result = msg.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text")
        if result.get("isError"):
            raise _classify(name, self._scrub(text))
        return text

    def budget_id(self) -> str:
        if self._budget_id is None:
            self._budget_id = parse_budget_id(self.call_tool("wallet_get_agent_info", {}))
        return self._budget_id

    def pay(self, url: str, idempotency_key: str) -> PaidResponse:
        args = {"url": url, "idempotency_key": idempotency_key, "budget_id": self.budget_id()}
        for attempt in itertools.count():
            try:
                text = self.call_tool("wallet_pay_url", args)
                break
            except TransientError as exc:
                if attempt >= self.retries:
                    raise
                logger.warning("wallet_pay_url %s: %s; retrying with the same idempotency key",
                               url, exc)
                time.sleep(self.backoff * (attempt + 1))
        if text.startswith("Payment requires approval"):
            txid = re.search(r"Transaction ID:\s*(\S+)", text)
            raise PaymentError("over the budget cap, waiting for an approval in Armada"
                               + (f" (transaction {txid.group(1)})" if txid else ""))
        txid, status, body, replayed = parse_completed(text)
        if status >= 400:
            raise PaymentError(f"merchant answered HTTP {status}")
        usd = self.paid_usd(txid) if txid else Decimal(0)   # no transaction: nothing was paid
        logger.info("paid %s via Armada: transaction %s, $%s%s", url, txid, usd,
                    " (idempotent replay)" if replayed else "")
        return PaidResponse(url, idempotency_key, txid, status, body, usd, replayed)

    def paid_usd(self, transaction_id: str, pages: int = 3) -> Decimal | None:
        """What Armada paid for ``transaction_id``, from ``wallet_list_transactions``.

        ``wallet_pay_url`` doesn't say what it paid, so read the transaction's
        row (listed as soon as it is signed). The list has no filter, so page
        through it. None if the row isn't found or isn't in USDC.
        """
        args: dict = {"limit": "100"}
        for _ in range(pages):
            try:
                text = self.call_tool("wallet_list_transactions", args)
            except PaymentError as exc:
                logger.warning("could not read the amount of transaction %s: %s", transaction_id, exc)
                return None
            usd = parse_transaction_amount(text, transaction_id)
            if usd is not None:
                return usd
            more = re.search(r'pass cursor="([^"]+)"', text)
            if not more:
                break
            args = {"limit": "100", "cursor": more.group(1)}
        logger.warning("transaction %s is not in the agent's transaction list", transaction_id)
        return None


def _jsonrpc_message(resp: requests.Response) -> dict:
    ctype = resp.headers.get("Content-Type", "")
    if "text/event-stream" in ctype:
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip())
                if "result" in msg or "error" in msg:
                    return msg
        raise PaymentError("Armada's event stream had no result")
    return resp.json()


_TRANSIENT_HINTS = ("try again", "upstream", "timed out", "timeout", "temporarily", "interrupted")


def _classify(name: str, text: str) -> PaymentError:
    first = text.strip().split("\n", 1)[0][:300]
    if any(h in text.lower() for h in _TRANSIENT_HINTS) and "do not retry" not in text.lower():
        return TransientError(f"{name}: {first}")
    return PaymentError(f"{name}: {first}")


def parse_transaction_amount(listing: str, transaction_id: str) -> Decimal | None:
    """The USDC amount on ``transaction_id``'s row of ``wallet_list_transactions``.

    Rows look like ``[2026-10-05 15:00] id=<id> debit 0.01 USDC [base · Mainnet] — x402 payment (completed)``.
    """
    m = re.search(rf"id={re.escape(transaction_id)} \S+ (\S+) (\S+)", listing)
    if not m or m.group(2) != "USDC":
        return None
    try:
        return Decimal(m.group(1))
    except InvalidOperation:
        return None
