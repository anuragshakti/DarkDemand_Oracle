"""
DarkDemand Oracle — Start
=========================
ONE command, ONE process, ONE port.

Usage:
    python run_pipeline.py

Opens: http://localhost:8501
  /               → Home (animated landing)
  /Signal_Scanner → Engine 1 · Dark Demand Intelligence
  /ML_Forecast    → Engine 2 · AI Rent/Yield Forecasting

Credentials:
    Update Senti_config.py → JLL_API section before each session.
    Tokens expire ~1 hour; jll_gpt.py auto-refreshes during the run.
"""

import subprocess, sys, os, time, webbrowser, signal
from pathlib import Path

BASE = Path(__file__).parent

def main():
    print()
    print("  ══════════════════════════════════════════════")
    print("    DarkDemand Oracle — Starting")
    print("    2026 JLL Hackathon · APAC")
    print("  ══════════════════════════════════════════════")
    print()
    print(f"  Directory : {BASE}")
    print()

    # Quick sanity check on credentials
    try:
        sys.path.insert(0, str(BASE))
        import Senti_config as cfg
        expiry_ms = cfg.JLL_API.get("token_expiry_ms", 0)
        import time as _t
        if expiry_ms < _t.time() * 1000:
            print("  ⚠️   Senti_config.py: JLL token looks expired — update before running Engine 1")
        else:
            print("  ✓   Senti_config.py: credentials present")
    except Exception as e:
        print(f"  ⚠️   Could not read Senti_config.py: {e}")
    print()

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(BASE / "home.py"),
            "--server.port",              "8501",
            "--server.headless",          "true",
            "--browser.gatherUsageStats", "false",
            "--server.runOnSave",         "false",
        ],
        cwd=str(BASE),
    )

    print("  ✓   http://localhost:8501           → Home")
    print("  ✓   http://localhost:8501/Signal_Scanner  → Engine 1")
    print("  ✓   http://localhost:8501/ML_Forecast     → Engine 2")
    print()
    print("  Opening browser in 3 seconds...")
    print("  Ctrl+C to stop.")
    print()

    time.sleep(3)
    webbrowser.open("http://localhost:8501")

    def _shutdown(sig, frame):
        print("\n  Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    proc.wait()

if __name__ == "__main__":
    main()
