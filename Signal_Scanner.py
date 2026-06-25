"""
Engine 1 — Dark Demand Intelligence
Thin wrapper: runs Senti_app.py in the correct directory context.
runpy.run_path sets __file__ correctly so all relative DB/model paths work.
"""
import sys, runpy
from pathlib import Path

ROOT = Path(__file__).parent.parent          # project root (one level up from pages/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "Senti_app.py"), run_name="__main__")
