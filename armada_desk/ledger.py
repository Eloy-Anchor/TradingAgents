"""Cost ledger for a desk run through Armada's inference proxy.

A LangChain callback that records every model call's team, role, model, and
token usage, and prices it from the public Armada model list
(``GET /v1/models``, prices in USD per 1M tokens). The proxy doesn't return a
price per call, so the totals are an estimate; check them against the agent's
spend on its Armada page.

It also carries a spend guard: once the estimated total reaches ``max_usd``,
the next model call raises ``SpendCapReached`` before it is sent.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# LangGraph node name -> (team, role). Matches tradingagents.graph.team_llms.TEAMS.
NODE_ROLES: dict[str, tuple[str, str]] = {
    "Market Analyst": ("analysts", "market"),
    "Sentiment Analyst": ("analysts", "social"),
    "News Analyst": ("analysts", "news"),
    "Fundamentals Analyst": ("analysts", "fundamentals"),
    "Bull Researcher": ("research", "bull"),
    "Bear Researcher": ("research", "bear"),
    "Research Manager": ("research", "research_manager"),
    "Trader": ("trader", "trader"),
    "Aggressive Analyst": ("risk", "aggressive"),
    "Neutral Analyst": ("risk", "neutral"),
    "Conservative Analyst": ("risk", "conservative"),
    "Portfolio Manager": ("portfolio", "portfolio_manager"),
}

# Calls made outside the graph come from the reflector, which bills with the
# portfolio team.
OUT_OF_GRAPH = ("portfolio", "reflector")

MILLION = Decimal(1_000_000)


class SpendCapReached(RuntimeError):
    pass


@dataclass
class CallRecord:
    team: str
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    usd: str


def _price_tier(entry: dict, input_tokens: int) -> dict:
    """The pricing that applies at this input length (long-context tiers)."""
    pricing = entry["pricing"]
    best = None
    for tier in entry.get("long_context_pricing") or []:
        threshold = tier["threshold"]
        applies = input_tokens >= threshold if tier.get("inclusive") else input_tokens > threshold
        if applies and (best is None or threshold > best["threshold"]):
            best = tier
    return best["pricing"] if best else pricing


def price_call(entry: dict, input_tokens: int, output_tokens: int,
               cache_read: int = 0, cache_write: int = 0) -> Decimal:
    """USD for one call, from a model-list entry."""
    p = _price_tier(entry, input_tokens)

    def rate(name: str, fallback: str = "input") -> Decimal:
        return Decimal(str(p.get(name, p.get(fallback, "0"))))

    uncached = max(input_tokens - cache_read - cache_write, 0)
    total = (
        uncached * rate("input")
        + cache_read * rate("cache_read")
        + cache_write * rate("cache_write")
        + output_tokens * rate("output", "output")
    )
    return total / MILLION


def _usage(response) -> tuple[int, int, int, int]:
    for gens in response.generations:
        for gen in gens:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                details = meta.get("input_token_details") or {}
                return (
                    int(meta.get("input_tokens") or 0),
                    int(meta.get("output_tokens") or 0),
                    int(details.get("cache_read") or 0),
                    int(details.get("cache_creation") or 0),
                )
    usage = (response.llm_output or {}).get("token_usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), 0, 0


class CostLedger(BaseCallbackHandler):
    """Records and prices every chat-model call; enforces ``max_usd``."""

    raise_error = True  # let SpendCapReached stop the run

    def __init__(self, models: dict[str, dict], max_usd: Decimal | None = None):
        self.models = models
        self.max_usd = max_usd
        self.calls: list[CallRecord] = []
        self.unpriced: list[str] = []
        self._pending: dict[Any, tuple[str, str, str]] = {}
        self._total = Decimal(0)
        self._lock = threading.Lock()

    @property
    def total_usd(self) -> Decimal:
        return self._total

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None,
                            invocation_params=None, **kwargs):
        with self._lock:
            if self.max_usd is not None and self._total >= self.max_usd:
                raise SpendCapReached(
                    f"estimated spend ${self._total:.4f} reached the ${self.max_usd} cap"
                )
            node = (metadata or {}).get("langgraph_node")
            team, role = NODE_ROLES.get(node, ("unknown", node)) if node else OUT_OF_GRAPH
            params = invocation_params or {}
            model = params.get("model") or params.get("model_name") or "unknown"
            self._pending[run_id] = (team, role, model)

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            team, role, model = self._pending.pop(run_id, ("unknown", "unknown", "unknown"))
            inp, out, cread, cwrite = _usage(response)
            entry = self.models.get(model)
            if entry is None:
                self.unpriced.append(model)
                usd = Decimal(0)
            else:
                usd = price_call(entry, inp, out, cread, cwrite)
            self._total += usd
            self.calls.append(CallRecord(team, role, model, inp, out, cread, cwrite, str(usd)))

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            self._pending.pop(run_id, None)

    def summary(self) -> dict:
        def roll(key: str) -> dict:
            out: dict[str, dict] = {}
            for c in self.calls:
                row = out.setdefault(getattr(c, key), {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": Decimal(0)})
                row["calls"] += 1
                row["input_tokens"] += c.input_tokens
                row["output_tokens"] += c.output_tokens
                row["usd"] += Decimal(c.usd)
            return {k: {**v, "usd": str(v["usd"])}
                    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["usd"])}

        return {
            "total_usd": str(self._total),
            "calls": len(self.calls),
            "by_team": roll("team"),
            "by_role": roll("role"),
            "unpriced_models": sorted(set(self.unpriced)),
            "records": [asdict(c) for c in self.calls],
        }
