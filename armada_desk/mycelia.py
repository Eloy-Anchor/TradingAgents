"""``get_signed_market_state``: the market analyst buys signed BTC data from Mycelia Signal.

Three endpoints, paid over x402 through Armada with the analysts' agent key,
about $0.11 a run:

- ``/oracle/price/btc/usd`` ($0.01), the median BTC/USD price across about 10 exchanges
- ``/oracle/sentiment/btc/usd`` ($0.05), MSXI, a derivatives sentiment index with a regime
- ``/oracle/stress/market`` ($0.05), MSSI, a market stress index with a regime

The market analyst doesn't decide whether to buy: in the first iteration-2
run it was offered the tool and never called it. So with ``registered`` the
analyst buys on its first turn, before the model runs, and the result goes into
its context with one line telling it to cite the values and regimes. The tool
stays on offer and returns the same answer for free.

Each response is saved as Mycelia sent it under ``<run dir>/mycelia/``, so
``python -m armada_desk.verify <run dir>`` can check its Ed25519 signature
later, and each payment goes into the ledger under analysts at the amount
Armada paid. The tool buys once per run: later calls get the same answer and
pay nothing.

The data is live, so the tool only buys when the run's trade date is today
(local date, as the runner's ``--date`` default); for any other date it says
so and pays nothing.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from armada_desk.ledger import CostLedger
from armada_desk.payer import ArmadaPayer, PaymentError
from tradingagents.agents.analysts import market_analyst
from tradingagents.graph import setup as graph_setup
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

logger = logging.getLogger(__name__)

BASE = "https://api.myceliasignal.com"
MERCHANT = "Mycelia Signal"
TEAM = "analysts"
ROLE = "mycelia"
TOOL_NAME = "get_signed_market_state"


@dataclass(frozen=True)
class Endpoint:
    slug: str      # file name under mycelia/, and part of the idempotency key
    path: str
    label: str


ENDPOINTS = (
    Endpoint("price-btc-usd", "/oracle/price/btc/usd", "BTC/USD price"),
    Endpoint("sentiment-btc-usd", "/oracle/sentiment/btc/usd", "MSXI sentiment"),
    Endpoint("stress-market", "/oracle/stress/market", "MSSI market stress"),
)

SCALES = {
    "sentiment-btc-usd": "-100 to +100; regimes EXTREMEBULLISH, BULLISH, NEUTRAL, BEARISH, EXTREMEBEARISH",
    "stress-market": "0 to 100; regimes CALM, ELEVATED, HIGH, EXTREME",
}


def regime_of(body: dict) -> str | None:
    """The regime label, from the body or, failing that, the signed canonical."""
    if body.get("regime"):
        return str(body["regime"])
    for field in str(body.get("canonical", "")).split("|"):
        if field.startswith("REGIME:"):
            return field.split(":", 1)[1]
    return None


def _when(body: dict) -> str:
    ts = body.get("timestamp") or body.get("computedAt")
    try:
        return datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return "time not given"


def describe(ep: Endpoint, body: dict) -> str:
    if ep.slug == "price-btc-usd":
        sources = body.get("sources") or []
        return (f"- {ep.label}: ${float(body['price']):,.2f}, {body.get('method', 'median')} of "
                f"{len(sources)} exchanges ({', '.join(sources)}), as of {_when(body)}")
    conf = body.get("confidence")
    conf = f", confidence {conf}" if conf is not None else ""
    return (f"- {ep.label}: {body.get('value')} (scale {SCALES[ep.slug]}), "
            f"regime {regime_of(body) or 'not given'}{conf}, as of {_when(body)}")


class SignedMarketState:
    """Buys, saves, and summarizes the three Mycelia responses for one run."""

    def __init__(self, payer: ArmadaPayer, run_dir: Path, ledger: CostLedger | None,
                 run_id: str, today: str | None = None):
        self.payer = payer
        self.dir = Path(run_dir) / "mycelia"
        self.ledger = ledger
        self.run_id = run_id
        self.today = today
        self._answer: str | None = None
        self._lock = threading.Lock()

    def _today(self) -> str:
        return self.today or date.today().isoformat()

    def __call__(self, trade_date: str) -> str:
        with self._lock:
            if self._answer is None:
                if trade_date and trade_date != self._today():
                    return (f"Mycelia Signal sells live data only; the trade date {trade_date} "
                            f"is not today ({self._today()}), so nothing was bought.")
                self._answer = self._buy()
            return self._answer

    def _buy(self) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        lines, paid = [], []
        for ep in ENDPOINTS:
            url = BASE + ep.path
            try:
                result = self.payer.pay(url, idempotency_key=f"desk-{self.run_id}-mycelia-{ep.slug}")
            except PaymentError as exc:
                lines.append(f"- {ep.label}: not available ({exc})")
                continue
            if self.ledger is not None:
                self.ledger.record_payment(TEAM, ROLE, MERCHANT, url, result.usd,
                                           result.transaction_id)
            paid.append(result.usd or Decimal(0))
            try:
                body = json.loads(result.body)
            except ValueError:
                lines.append(f"- {ep.label}: paid, but Mycelia's answer was not JSON")
                continue
            (self.dir / f"{ep.slug}.json").write_text(result.body)
            try:
                lines.append(describe(ep, body))
            except (KeyError, TypeError, ValueError):
                lines.append(f"- {ep.label}: paid and saved, but not in the expected shape: "
                             f"{result.body[:300]}")
        total = sum(paid, Decimal(0))
        return "\n".join([
            f"Mycelia Signal, bought just now for ${total} through Armada. Each value is "
            "Ed25519-signed by Mycelia, and the signed responses are saved with this run "
            "so anyone can check later what the desk saw.",
            *lines,
            "Use these as live context next to the price history: the regimes are Mycelia's "
            "own classifications, and a value's age is in its 'as of'.",
        ])


INSTRUCTION = (
    "Mycelia Signal, a paid source, sold the desk this signed, live BTC market state for this "
    "run (get_signed_market_state returns the same text at no extra cost). Cite these values and "
    "regimes in your report, and say where they agree or disagree with the price history and "
    "indicators:\n\n{data}"
)


def _read(state: SignedMarketState, trade_date: str) -> str:
    try:
        return state(trade_date)
    except Exception as exc:  # an exception out of a tool or node would end the run
        logger.warning("%s failed: %s", TOOL_NAME, exc)
        return f"Mycelia Signal is unavailable for this run ({type(exc).__name__})."


def with_market_state(create_analyst, state: SignedMarketState):
    """Wrap the market analyst's factory so each of its turns starts with the Mycelia data.

    The first turn buys it; later turns reuse that answer. The note goes right
    after the run's opening message, at the same place every turn, so the
    analyst's own tool calls and results follow it as before.
    """
    def create(llm):
        node = create_analyst(llm)

        def market_analyst_node(graph_state):
            note = HumanMessage(INSTRUCTION.format(data=_read(state, graph_state.get("trade_date", ""))))
            messages = list(graph_state["messages"])
            return node({**graph_state, "messages": [*messages[:1], note, *messages[1:]]})

        return market_analyst_node

    return create


def make_tool(state: SignedMarketState):
    @tool(TOOL_NAME)
    def get_signed_market_state(
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """Live, cryptographically signed BTC market state from Mycelia Signal (a paid source).

        Returns the BTC/USD median price across about 10 exchanges, the MSXI
        derivatives sentiment index with its regime label, and the MSSI market
        stress index with its regime label. Bought once per run; calling it
        again returns the same answer at no extra cost.
        """
        return _read(state, trade_date)

    return get_signed_market_state


@contextmanager
def registered(state: SignedMarketState):
    """Give the market analyst the Mycelia data and tool while the block runs.

    The market analyst reads its tools from ``market_analyst.TOOLS`` on every
    turn, its tool node is built from the analyst's spec, and its node from
    ``create_market_analyst`` when the graph is set up, so all three change for
    the length of the block, graph setup and run.
    """
    spec = ANALYST_NODE_SPECS["market"]
    tools = market_analyst.TOOLS
    create = graph_setup.create_market_analyst
    market_analyst.TOOLS = (*tools, make_tool(state))
    ANALYST_NODE_SPECS["market"] = replace(spec, tools=market_analyst.TOOLS)
    graph_setup.create_market_analyst = with_market_state(create, state)
    try:
        yield
    finally:
        market_analyst.TOOLS = tools
        ANALYST_NODE_SPECS["market"] = spec
        graph_setup.create_market_analyst = create
