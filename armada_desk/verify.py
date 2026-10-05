"""Verify the signed Mycelia Signal responses a desk run bought.

    python -m armada_desk.verify <run dir>

Checks every ``<run dir>/mycelia/*.json`` against Mycelia's published Ed25519
keys (``GET /keys/ed25519``): the response's ``pubkey`` must be one of them,
and its ``signature`` must verify over the UTF-8 bytes of its ``canonical``
string (RFC 8032, no pre-hash). Prints one line per file and exits 0 only when
every signature verifies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

KEYS_URL = "https://api.myceliasignal.com/keys/ed25519"


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def published_keys(doc: dict) -> dict[str, str]:
    """pubkey hex -> node name, from the /keys/ed25519 document."""
    keys = {k["pubkey"].lower(): k.get("node", "?") for k in doc.get("pubkeys") or []}
    if doc.get("pubkey"):
        keys.setdefault(doc["pubkey"].lower(), "default")
    return keys


def fetch_keys(url: str = KEYS_URL) -> dict[str, str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return published_keys(resp.json())


def verify_response(body: dict, keys: dict[str, str]) -> tuple[bool, str]:
    """(verified, why) for one signed Mycelia response."""
    canonical, signature, pubkey = body.get("canonical"), body.get("signature"), body.get("pubkey")
    if not (canonical and signature and pubkey):
        return False, "missing canonical, signature, or pubkey"
    if body.get("signingScheme", "ed25519") != "ed25519":
        return False, f"unexpected signing scheme {body['signingScheme']!r}"
    node = keys.get(pubkey.lower())
    if node is None:
        return False, f"pubkey {pubkey[:16]}... is not a published Mycelia key"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey)).verify(
            bytes.fromhex(signature), canonical.encode("utf-8"))
    except (InvalidSignature, ValueError):
        return False, "signature does not match canonical"
    return True, f"signed by the {node} key"


def verify_dir(run_dir: Path, keys: dict[str, str]) -> list[Result]:
    results = []
    for path in sorted((run_dir / "mycelia").glob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            results.append(Result(path.name, False, f"unreadable: {exc}"))
            continue
        ok, why = verify_response(body, keys)
        results.append(Result(path.name, ok, f"{why}: {body.get('canonical', '')}"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--keys", type=Path, default=None,
                    help=f"a saved copy of {KEYS_URL} (default: fetch it)")
    args = ap.parse_args(argv)

    keys = published_keys(json.loads(args.keys.read_text())) if args.keys else fetch_keys()
    results = verify_dir(args.run_dir, keys)
    for r in results:
        print(f"{'VERIFIED' if r.ok else 'FAILED  '}  {r.name}  {r.detail}")
    passed = sum(r.ok for r in results)
    if not results:
        print(f"no signed responses under {args.run_dir / 'mycelia'}")
        return 1
    print(f"{passed}/{len(results)} signatures verified against {args.keys or KEYS_URL}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
