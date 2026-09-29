"""The Armada desk's cost ledger: per-team pricing and the spend guard."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from armada_desk.ledger import CostLedger, SpendCapReached, price_call
from tests.test_graph_end_to_end import offline  # noqa: F401,F811  (fixture)

MODELS = {
    "quick": {"pricing": {"input": "0.1", "output": "0.5", "cache_read": "0.01"}},
    "deep": {
        "pricing": {"input": "2", "output": "10"},
        "long_context_pricing": [
            {"threshold": 1000, "inclusive": False, "pricing": {"input": "4", "output": "15"}},
        ],
    },
}


def _call(ledger, node, model, inp, out, cache_read=0):
    run_id = uuid.uuid4()
    metadata = {"langgraph_node": node} if node else {}
    ledger.on_chat_model_start({}, [[]], run_id=run_id, metadata=metadata,
                               invocation_params={"model": model})
    msg = AIMessage(content="x", usage_metadata={
        "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
        "input_token_details": {"cache_read": cache_read},
    })
    ledger.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run_id)


@pytest.mark.unit
def test_ledger_prices_usage_by_team():
    ledger = CostLedger(MODELS)

    # analysts: 1000 in (200 cached) + 100 out on quick
    _call(ledger, "Market Analyst", "quick", 1000, 100, cache_read=200)
    # research: 1000 in at base price, 1001 in on the long-context tier
    _call(ledger, "Research Manager", "deep", 1000, 10)
    _call(ledger, "Research Manager", "deep", 1001, 10)
    # reflector runs outside the graph and bills with portfolio
    _call(ledger, None, "quick", 100, 10)

    s = ledger.summary()
    analysts = (Decimal(800) * Decimal("0.1") + 200 * Decimal("0.01") + 100 * Decimal("0.5")) / 1_000_000
    research = (1000 * 2 + 10 * 10 + 1001 * 4 + 10 * 15) / Decimal(1_000_000)
    portfolio = (100 * Decimal("0.1") + 10 * Decimal("0.5")) / 1_000_000
    assert Decimal(s["by_team"]["analysts"]["usd"]) == analysts
    assert Decimal(s["by_team"]["research"]["usd"]) == research
    assert Decimal(s["by_team"]["portfolio"]["usd"]) == portfolio
    assert s["by_role"]["reflector"]["calls"] == 1
    assert Decimal(s["total_usd"]) == analysts + research + portfolio
    assert list(s["by_team"]) == ["research", "analysts", "portfolio"]


@pytest.mark.unit
def test_long_context_inclusive_threshold():
    entry = {"pricing": {"input": "1", "output": "1"},
             "long_context_pricing": [{"threshold": 10, "inclusive": True,
                                       "pricing": {"input": "2", "output": "2"}}]}
    assert price_call(entry, 10, 0) == Decimal(20) / 1_000_000
    assert price_call(entry, 9, 0) == Decimal(9) / 1_000_000


@pytest.mark.unit
def test_unpriced_model_is_reported():
    ledger = CostLedger(MODELS)
    _call(ledger, "Trader", "not-listed", 10, 10)
    assert ledger.summary()["unpriced_models"] == ["not-listed"]


@pytest.mark.unit
def test_spend_guard_stops_before_next_call():
    ledger = CostLedger(MODELS, max_usd=Decimal("0.002"))
    _call(ledger, "Research Manager", "deep", 1000, 0)   # $0.002, reaches the cap

    with pytest.raises(SpendCapReached):
        _call(ledger, "Trader", "quick", 10, 10)
    assert len(ledger.calls) == 1


@pytest.mark.unit
def test_ledger_attributes_calls_in_a_full_run(tmp_path, monkeypatch, offline):  # noqa: F811
    """LangGraph's node name reaches the ledger through the model's callbacks."""
    from tests import test_graph_end_to_end as e2e

    class PricedModel(e2e.ScriptedModel):
        model: str = "quick"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop, run_manager, **kwargs)
            result.generations[0].message.usage_metadata = {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
            return result

    ledger = CostLedger(MODELS)
    cfg = dict(e2e.DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"),
               memory_log_path=str(tmp_path / "log.md"))
    monkeypatch.setattr(e2e.trading_graph, "create_llm_client",
                        lambda **k: e2e._Client(PricedModel(callbacks=k.get("callbacks"))))
    graph = e2e.trading_graph.TradingAgentsGraph(config=cfg, callbacks=[ledger])
    graph.propagate("NVDA", e2e.TRADE_DATE)

    teams = ledger.summary()["by_team"]
    assert set(teams) == {"analysts", "research", "trader", "risk", "portfolio"}
    assert "unknown" not in teams
