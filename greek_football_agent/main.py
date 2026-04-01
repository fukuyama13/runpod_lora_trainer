"""
Entry point: run the Greek football digest agent and save output to disk.
"""

from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv

from agent import run_agent


def main() -> None:
    load_dotenv()
    digest = run_agent()
    print(digest)
    print()

    out_dir = os.path.join(os.path.dirname(__file__), "digest_output")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"digest_{date.today().isoformat()}.txt"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(digest)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
