"""Run the TradingAgents desk through Armada's inference proxy and price it.

Each team (analysts, research, trader, risk, portfolio) reads its Armada agent
key from its own environment variable, so each team bills its own agent. Keys
are only ever read from the environment, never written anywhere.

    ARMADA_AGENT_KEY=... python -m armada_desk.run --ticker BTC-USD --max-usd 3

With only ``--key-env`` every team uses that one key. ``--team-key-env
research=RESEARCH_KEY`` (repeatable) gives a team its own key.

Prints one JSON line with the decision and the estimated cost by team and role,
and writes the same summary plus every priced call to ``<out>/ledger.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import requests

from armada_desk.ledger import CostLedger, SpendCapReached
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.team_llms import TEAMS
from tradingagents.graph.trading_graph import TradingAgentsGraph

BASE_URL = "https://inference.agentic.anchorage.com/v1"
QUICK = "openai/gpt-6-luna"
DEEP = "openai/gpt-6-sol"
# Fundamentals drops out for crypto.
ANALYSTS = ("market", "social", "news")


def fetch_models(base_url: str) -> dict[str, dict]:
    resp = requests.get(f"{base_url}/models", timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return {m["id"]: m for m in body.get("data", body) if m.get("pricing")}


def team_key_envs(default_env: str, overrides: list[str]) -> dict[str, str]:
    envs = dict.fromkeys(TEAMS, default_env)
    for item in overrides:
        team, _, env = item.partition("=")
        if team not in TEAMS or not env:
            raise SystemExit(f"--team-key-env wants TEAM=ENV_VAR with TEAM in {', '.join(TEAMS)}")
        envs[team] = env
    return envs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticker", default="BTC-USD")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--max-usd", type=Decimal, required=True,
                    help="stop before the next call once the estimated spend reaches this")
    ap.add_argument("--key-env", default="ARMADA_AGENT_KEY")
    ap.add_argument("--team-key-env", action="append", default=[])
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--quick", default=QUICK)
    ap.add_argument("--deep", default=DEEP)
    ap.add_argument("--out", default=None, help="run directory (default ~/.tradingagents/armada/<ts>)")
    args = ap.parse_args(argv)

    models = fetch_models(args.base_url)
    for m in (args.quick, args.deep):
        if m not in models:
            raise SystemExit(f"model {m} is not in {args.base_url}/models")

    out = Path(args.out or Path.home() / ".tradingagents" / "armada" / time.strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    envs = team_key_envs(args.key_env, args.team_key_env)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        llm_provider="openai_compatible",
        backend_url=args.base_url,
        quick_think_llm=args.quick,
        deep_think_llm=args.deep,
        team_llms={team: {"api_key_env": env} for team, env in envs.items()},
        results_dir=str(out / "results"),
        # A fresh memory log per run keeps runs independent: no reflection on
        # earlier runs' decisions, so every run makes the same kind of calls.
        memory_log_path=str(out / "memory.md"),
    )

    ledger = CostLedger(models, max_usd=args.max_usd)
    graph = TradingAgentsGraph(selected_analysts=ANALYSTS, config=cfg, callbacks=[ledger])

    started = time.monotonic()
    status, signal, error = "ok", None, None
    try:
        _, signal = graph.propagate(args.ticker, args.date, asset_type="crypto")
    except SpendCapReached as exc:
        status, error = "spend_cap_reached", str(exc)
    except Exception as exc:  # report what was spent before the failure
        status, error = "error", f"{type(exc).__name__}: {exc}"

    summary = {
        "status": status,
        "ticker": args.ticker,
        "date": args.date,
        "signal": signal,
        "error": error,
        "seconds": round(time.monotonic() - started, 1),
        "models": {"quick": args.quick, "deep": args.deep},
        "max_usd": str(args.max_usd),
        "out": str(out),
        **ledger.summary(),
    }
    (out / "ledger.json").write_text(json.dumps(summary, indent=2))
    printable = {k: v for k, v in summary.items() if k != "records"}
    print(json.dumps(printable))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
