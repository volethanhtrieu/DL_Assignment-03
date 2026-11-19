import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M2M3_DIR = SCRIPT_DIR.parent

if str(M2M3_DIR) not in sys.path:
    sys.path.insert(0, str(M2M3_DIR))

DEFAULT_ROOT = SCRIPT_DIR.parents[2] / "M3_Local_Input"
DATA_ROOT = Path(
    os.environ.get("CO3133_M3_DATA_ROOT") or DEFAULT_ROOT
).expanduser().resolve()
