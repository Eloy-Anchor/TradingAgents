"""Mycelia Signal through Armada: the payer, the tool, the verify step, the runner flag.

No network: Armada's /mcp is a fake session, and Mycelia's answers are saved
responses (one paid, two free previews), all genuinely signed.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest
import requests

from armada_desk import mycelia, run, verify
from armada_desk.ledger import CostLedger
from armada_desk.payer import ArmadaPayer, PaidResponse, PaymentError, parse_transaction_amount
from tests.test_graph_end_to_end import offline  # noqa: F401,F811  (fixture)
from tradingagents.agents.analysts import market_analyst
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

FIXTURES = Path(__file__).parent / "fixtures" / "mycelia"
KEY = "agentic_ak_test_not_a_real_key_0123456789"
KEY_ENV = "DESK_TEST_ANALYSTS_KEY"
BUDGET = "bud-analysts-1"

AGENT_INFO = (
    "Agent: Desk Analyst\nNetwork: Mainnet\n\n"
    "Budgets (admissible for this agent — pass one of these ids as budget_id to wallet_pay_url):\n"
    "Effective spend limit: $2 (all_time) — $1.99 remaining\n"
    f"Spend against → budget_id: {BUDGET}\n"
)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


BODIES = {
    "/oracle/price/btc/usd": _fixture("price-btc-usd.json"),
    "/oracle/sentiment/btc/usd": _fixture("sentiment-btc-usd-preview.json"),
    "/oracle/stress/market": _fixture("stress-market-preview.json"),
}
PRICES = {"/oracle/price/btc/usd": "0.01", "/oracle/sentiment/btc/usd": "0.05",
          "/oracle/stress/market": "0.05"}


def _completed(txid: str, body: str) -> str:
    return (f"Payment completed (protocol v2).\nTransaction ID: {txid}\nHTTP 200\n"
            'Settlement (PAYMENT-RESPONSE): {"success":true,"transaction":"0xabc"}\n' + body)


class _Resp:
    def __init__(self, status: int, result: dict | None = None, sse: bool = True):
        self.status_code = status
        msg = {"jsonrpc": "2.0", "id": 1, "result": result or {}}
        self.headers = {"Content-Type": "text/event-stream" if sse else "application/json"}
        self.text = f"event: message\ndata: {json.dumps(msg)}\n\n" if sse else json.dumps(msg)

    def json(self):
        return json.loads(self.text)


def _text(text: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": error}


class FakeArmada:
    """Answers tools/call like Armada's /mcp; ``script`` overrides per call."""

    def __init__(self, script=None):
        self.posts: list[dict] = []
        self.script = list(script or [])
        self.txs: dict[str, str] = {}

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if step is not None:
                return step
        name, args = json["params"]["name"], json["params"]["arguments"]
        if name == "wallet_get_agent_info":
            return _Resp(200, _text(AGENT_INFO))
        if name == "wallet_pay_url":
            path = args["url"].removeprefix(mycelia.BASE)
            txid = f"tx-{len(self.txs) + 1}"
            self.txs[txid] = PRICES[path]
            return _Resp(200, _text(_completed(txid, BODIES[path])))
        if name == "wallet_list_transactions":
            rows = "".join(f"[2026-10-05 15:00] id={t} debit {a} USDC [base · Mainnet] — "
                           "x402 payment (pending_confirmation)\n" for t, a in self.txs.items())
            return _Resp(200, _text(rows or "No transactions found."))
        raise AssertionError(name)

    def calls(self, name: str) -> list[dict]:
        return [p["json"]["params"]["arguments"] for p in self.posts
                if p["json"]["params"]["name"] == name]


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    return KEY


def _payer(armada: FakeArmada) -> ArmadaPayer:
    return ArmadaPayer(KEY_ENV, mcp_url="https://armada.test/mcp", backoff=0, session=armada)


# -- the payer ---------------------------------------------------------------

@pytest.mark.unit
def test_payer_sends_the_wallet_pay_url_arguments(key):
    armada = FakeArmada()
    paid = _payer(armada).pay(mycelia.BASE + "/oracle/price/btc/usd", idempotency_key="desk-r1-x")

    assert armada.calls("wallet_pay_url") == [{
        "url": "https://api.myceliasignal.com/oracle/price/btc/usd",
        "idempotency_key": "desk-r1-x",
        "budget_id": BUDGET,   # from wallet_get_agent_info
    }]
    post = armada.posts[0]
    assert post["url"] == "https://armada.test/mcp"
    assert post["json"]["method"] == "tools/call"
    assert post["headers"]["Authorization"] == f"Bearer {KEY}"
    assert "text/event-stream" in post["headers"]["Accept"]
    assert paid.transaction_id == "tx-1" and paid.status == 200
    assert paid.body == BODIES["/oracle/price/btc/usd"]
    assert paid.usd == Decimal("0.01")   # read back from wallet_list_transactions


@pytest.mark.unit
def test_payer_never_puts_the_key_in_errors_or_logs(key, caplog):
    caplog.set_level(logging.DEBUG)
    leaky = requests.ConnectionError(f"failed with Authorization: Bearer {KEY}")
    armada = FakeArmada([None] + [leaky] * 3)   # agent info, then the pay call fails 3 times
    with pytest.raises(PaymentError) as exc:
        _payer(armada).pay(mycelia.BASE + "/oracle/price/btc/usd", idempotency_key="k")
    assert KEY not in str(exc.value)
    assert exc.value.__suppress_context__   # the leaky exception isn't chained into tracebacks

    rejected = FakeArmada([None, _Resp(200, _text(f"merchant not in budget allowlist ({KEY})", error=True))])
    with pytest.raises(PaymentError) as exc:
        _payer(rejected).pay(mycelia.BASE + "/oracle/stress/market", idempotency_key="k2")
    assert "allowlist" in str(exc.value) and KEY not in str(exc.value)

    assert KEY not in caplog.text
    assert len(rejected.calls("wallet_pay_url")) == 1   # a compliance reject isn't retried


@pytest.mark.unit
def test_payer_needs_the_key_env(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(PaymentError, match=KEY_ENV):
        _payer(FakeArmada()).budget_id()


@pytest.mark.unit
def test_a_retry_reuses_the_same_idempotency_key(key):
    armada = FakeArmada([None, _Resp(503), requests.Timeout("slow")])
    paid = _payer(armada).pay(mycelia.BASE + "/oracle/sentiment/btc/usd", idempotency_key="desk-r1-s")

    keys = [a["idempotency_key"] for a in armada.calls("wallet_pay_url")]
    assert keys == ["desk-r1-s"] * 3
    assert paid.usd == Decimal("0.05")


@pytest.mark.unit
def test_payer_reads_an_idempotent_replay_and_approval(key):
    replay = ('[IDEMPOTENCY CACHE HIT — idempotency_key "k" was first used at 2026-10-05T15:00:00Z.\n'
              " The response below is the SAME cached result. If your current request differs,\n"
              " use a new idempotency_key.]\n\n" + _completed("tx-9", '{"a":1}'))
    armada = FakeArmada([None, _Resp(200, _text(replay)),
                         _Resp(200, _text("[2026-10-05 15:00] id=tx-9 debit 0.05 USDC — x402 payment (completed)\n"))])
    paid = _payer(armada).pay("https://m.test/x", idempotency_key="k")
    assert paid.replayed and paid.body == '{"a":1}' and paid.usd == Decimal("0.05")

    approval = "Payment requires approval.\nTransaction ID: tx-7\nStatus: pending_approval\nThe human user must approve."
    with pytest.raises(PaymentError, match="approval.*tx-7"):
        _payer(FakeArmada([None, _Resp(200, _text(approval))])).pay("https://m.test/x", idempotency_key="k3")


@pytest.mark.unit
def test_transaction_amount_parsing():
    listing = ("[2026-10-05 15:00] id=tx-1 debit 0.01 USDC [base · Mainnet] — x402 payment (completed)\n"
               "[2026-10-05 15:00] id=tx-10 debit 0.05 USDC [base · Mainnet] — x402 payment (completed)\n")
    assert parse_transaction_amount(listing, "tx-1") == Decimal("0.01")
    assert parse_transaction_amount(listing, "tx-10") == Decimal("0.05")
    assert parse_transaction_amount(listing, "tx-2") is None


# -- the tool ----------------------------------------------------------------

class FakePayer:
    def __init__(self):
        self.keys: list[str] = []

    def pay(self, url, idempotency_key):
        self.keys.append(idempotency_key)
        path = url.removeprefix(mycelia.BASE)
        return PaidResponse(url, idempotency_key, f"tx-{len(self.keys)}", 200, BODIES[path],
                            Decimal(PRICES[path]))


@pytest.mark.unit
def test_tool_saves_the_signed_responses_and_returns_the_regimes(tmp_path):
    ledger, payer = CostLedger({}), FakePayer()
    state = mycelia.SignedMarketState(payer, tmp_path, ledger, run_id="r1", today="2026-10-05")
    tool = mycelia.make_tool(state)

    out = tool.func(trade_date="2026-10-05")

    assert "$85,633.84" in out and "10 exchanges" in out
    assert "regime NEUTRAL" in out and "regime CALM" in out
    assert "$0.11" in out
    for path, slug in (("/oracle/price/btc/usd", "price-btc-usd"),
                       ("/oracle/sentiment/btc/usd", "sentiment-btc-usd"),
                       ("/oracle/stress/market", "stress-market")):
        assert (tmp_path / "mycelia" / f"{slug}.json").read_text() == BODIES[path]
    assert payer.keys == ["desk-r1-mycelia-price-btc-usd", "desk-r1-mycelia-sentiment-btc-usd",
                          "desk-r1-mycelia-stress-market"]

    # A second call in the same run pays nothing.
    assert tool.func(trade_date="2026-10-05") == out
    assert len(payer.keys) == 3
    assert [p.usd for p in ledger.payments] == ["0.01", "0.05", "0.05"]


@pytest.mark.unit
def test_tool_buys_nothing_for_a_past_date(tmp_path):
    payer = FakePayer()
    state = mycelia.SignedMarketState(payer, tmp_path, None, run_id="r1", today="2026-10-05")
    assert "nothing was bought" in mycelia.make_tool(state).func(trade_date="2026-10-02")
    assert payer.keys == []


@pytest.mark.unit
def test_tool_reports_a_failed_purchase_and_keeps_the_rest(tmp_path):
    class Refused(FakePayer):
        def pay(self, url, idempotency_key):
            if "stress" in url:
                raise PaymentError("wallet_pay_url: merchant not in budget allowlist")
            return super().pay(url, idempotency_key)

    ledger = CostLedger({})
    state = mycelia.SignedMarketState(Refused(), tmp_path, ledger, run_id="r1", today="2026-10-05")
    out = mycelia.make_tool(state).func(trade_date="2026-10-05")
    assert "MSSI market stress: not available (wallet_pay_url: merchant not in budget allowlist)" in out
    assert "regime NEUTRAL" in out
    assert not (tmp_path / "mycelia" / "stress-market.json").exists()
    assert ledger.summary()["data_usd"] == "0.06"


@pytest.mark.unit
def test_regime_comes_from_the_canonical_when_the_body_has_none():
    assert mycelia.regime_of({"canonical": "v2|SENTIMENT|BTCUSD|MSXI|30.1|INDEX|FR:1:0.3|REGIME:BULLISH|CONFIDENCE:1|METHOD:v2|1|2"}) == "BULLISH"


# -- the ledger --------------------------------------------------------------

@pytest.mark.unit
def test_ledger_adds_the_paid_amount_under_analysts():
    ledger = CostLedger({})
    ledger.record_payment("analysts", "mycelia", "Mycelia Signal", "u1", Decimal("0.01"), "tx-1")
    ledger.record_payment("analysts", "mycelia", "Mycelia Signal", "u2", Decimal("0.05"), "tx-2")
    ledger.record_payment("analysts", "mycelia", "Mycelia Signal", "u3", None, "tx-3")

    s = ledger.summary()
    assert s["by_team"]["analysts"]["usd"] == "0.06"
    assert s["by_team"]["analysts"]["payments"] == 3
    assert s["by_role"]["mycelia"]["calls"] == 0
    assert s["total_usd"] == s["data_usd"] == "0.06"
    assert [p["amount_known"] for p in s["payments"]] == [True, True, False]


# -- verify ------------------------------------------------------------------

KEYS = verify.published_keys(json.loads(_fixture("keys-ed25519.json")))


@pytest.mark.unit
def test_verify_passes_the_known_good_response():
    ok, why = verify.verify_response(json.loads(_fixture("price-btc-usd.json")), KEYS)
    assert ok, why
    assert "us key" in why


@pytest.mark.unit
@pytest.mark.parametrize("field", ["canonical", "signature"])
def test_verify_fails_with_one_byte_flipped(field):
    body = json.loads(_fixture("price-btc-usd.json"))
    raw = bytearray(body[field].encode())
    raw[len(raw) // 2] ^= 0x01
    body[field] = raw.decode()
    ok, why = verify.verify_response(body, KEYS)
    assert not ok


@pytest.mark.unit
def test_verify_rejects_an_unpublished_key():
    body = json.loads(_fixture("price-btc-usd.json"))
    ok, why = verify.verify_response(body, {"00" * 32: "other"})
    assert not ok and "not a published" in why


@pytest.mark.unit
def test_verify_cli_over_a_run_dir(tmp_path, capsys):
    (tmp_path / "mycelia").mkdir()
    for name in ("price-btc-usd.json", "sentiment-btc-usd-preview.json", "stress-market-preview.json"):
        (tmp_path / "mycelia" / name).write_text(_fixture(name))
    keys = str(FIXTURES / "keys-ed25519.json")

    assert verify.main([str(tmp_path), "--keys", keys]) == 0
    assert "3/3 signatures verified" in capsys.readouterr().out

    tampered = _fixture("price-btc-usd.json").replace("85633.84", "95633.84")
    (tmp_path / "mycelia" / "price-btc-usd.json").write_text(tampered)
    assert verify.main([str(tmp_path), "--keys", keys]) == 1
    out = capsys.readouterr().out
    assert "FAILED    price-btc-usd.json" in out and "2/3 signatures verified" in out


# -- the runner flag ---------------------------------------------------------

def _run_main(monkeypatch, tmp_path, *flags) -> list[str]:
    """run.main with no network; returns the market analyst's tool names at graph setup."""
    seen: list[str] = []

    class Graph:
        def __init__(self, **kwargs):
            seen.extend(t.name for t in ANALYST_NODE_SPECS["market"].tools)
            assert ANALYST_NODE_SPECS["market"].tools == market_analyst.TOOLS

        def propagate(self, *a, **k):
            return {}, "Hold"

    monkeypatch.setattr(run, "fetch_models", lambda url: {run.QUICK: {}, run.DEEP: {}})
    monkeypatch.setattr(run, "TradingAgentsGraph", Graph)
    monkeypatch.setattr(run.ArmadaPayer, "budget_id", lambda self: BUDGET)
    assert run.main(["--max-usd", "1", "--out", str(tmp_path), "--key-env", KEY_ENV, *flags]) == 0
    return seen


@pytest.mark.unit
def test_no_mycelia_leaves_the_tool_out(monkeypatch, tmp_path, key):
    assert mycelia.TOOL_NAME not in _run_main(monkeypatch, tmp_path, "--no-mycelia")
    assert mycelia.TOOL_NAME not in _run_main(monkeypatch, tmp_path)   # the default
    assert json.loads((tmp_path / "ledger.json").read_text())["mycelia"] is False


@pytest.mark.unit
def test_mycelia_registers_the_tool_for_the_run_only(monkeypatch, tmp_path, key):
    before = market_analyst.TOOLS
    assert mycelia.TOOL_NAME in _run_main(monkeypatch, tmp_path, "--mycelia")
    assert market_analyst.TOOLS is before
    assert mycelia.TOOL_NAME not in [t.name for t in ANALYST_NODE_SPECS["market"].tools]


@pytest.mark.unit
def test_mycelia_needs_btc():
    with pytest.raises(SystemExit, match="BTC-USD"):
        run.main(["--max-usd", "1", "--ticker", "ETH-USD", "--mycelia"])


@pytest.mark.unit
def test_market_analyst_calls_the_tool_in_a_full_run(tmp_path, monkeypatch, offline):  # noqa: F811
    from tests import test_graph_end_to_end as e2e

    ledger, payer = CostLedger({}), FakePayer()
    state = mycelia.SignedMarketState(payer, tmp_path / "run", ledger, run_id="r1", today=e2e.TRADE_DATE)
    with mycelia.registered(state):
        graph = e2e._graph(tmp_path, monkeypatch, e2e.ScriptedModel())
        graph.propagate("NVDA", e2e.TRADE_DATE)

    assert len(payer.keys) == 3   # bought once, though the model also called the tool
    assert sorted(p.name for p in (tmp_path / "run" / "mycelia").iterdir()) == [
        "price-btc-usd.json", "sentiment-btc-usd.json", "stress-market.json"]
    assert ledger.summary()["by_team"]["analysts"]["usd"] == "0.11"


@pytest.mark.unit
def test_market_analyst_buys_and_sees_the_data_without_calling_the_tool(tmp_path, monkeypatch, offline):  # noqa: F811
    """Run B of iteration 2: the model never called the tool. The purchase can't depend on it."""
    from tests import test_graph_end_to_end as e2e

    prompts: list[str] = []

    class NoTools(e2e.ScriptedModel):
        def bind_tools(self, tools, **kwargs):
            return self   # answers straight away, never calls a tool

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompts.append("\n".join(str(m.content) for m in messages))
            return super()._generate(messages, stop, run_manager, **kwargs)

    ledger, payer = CostLedger({}), FakePayer()
    state = mycelia.SignedMarketState(payer, tmp_path / "run", ledger, run_id="r1", today=e2e.TRADE_DATE)
    with mycelia.registered(state):
        graph = e2e._graph(tmp_path, monkeypatch, NoTools())
        graph.propagate("NVDA", e2e.TRADE_DATE)

    assert len(payer.keys) == 3
    assert ledger.summary()["data_usd"] == "0.11"
    market = [p for p in prompts if "get_signed_market_state" in p and "Cite these values" in p]
    assert len(market) == 1   # only the market analyst gets it
    assert "$85,633.84" in market[0] and "regime NEUTRAL" in market[0] and "regime CALM" in market[0]


@pytest.mark.unit
def test_registered_restores_the_market_analyst():
    from tradingagents.graph import setup as graph_setup

    create, tools, spec = graph_setup.create_market_analyst, market_analyst.TOOLS, ANALYST_NODE_SPECS["market"]
    state = mycelia.SignedMarketState(FakePayer(), Path("/nonexistent"), None, run_id="r1")
    with mycelia.registered(state):
        assert graph_setup.create_market_analyst is not create
        assert market_analyst.TOOLS[-1].name == mycelia.TOOL_NAME
    assert (graph_setup.create_market_analyst, market_analyst.TOOLS, ANALYST_NODE_SPECS["market"]) == (create, tools, spec)
