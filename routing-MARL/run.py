"""Low-level runner: start routing_MARl/main.py with SUMO in-process (libsumo), which is faster than TraCI over TCP.

train.py and infer.py call this file; you rarely need it directly.
- `import traci` in the code is mapped to libsumo, so the code itself stays unchanged.
- PyTorch uses at most TORCH_THREADS CPU threads (default 8), so several runs can share one machine.
- pid.txt (at start) and done.txt (at the end: 'ok', or the error) are written next to the run's --output-dir folder.
  Trust done.txt, not the process exit code: on Windows, runs with the LSTM encoder can abort while the CUDA and
  libsumo libraries unload, after the run has finished and saved everything (exit code 0xC0000409).
- Set COOPEVA_TRACI=1 to use plain TraCI instead of libsumo (--gui does this by itself).

Usage (from routing-MARL): python run.py <main.py arguments...>
"""
import os
import runpy
import sys
import traceback
from pathlib import Path

import xml.parsers.expat  # noqa: F401  load the env's expat DLL before libsumo's
import matplotlib.pyplot  # noqa: F401
import torch

if os.environ.get("PROC_ABOVE_NORMAL") and os.name == "nt":  # opt-in on Windows: higher priority for this process only
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
_cap = int(os.environ.get("TORCH_THREADS", "8"))
_set_num_threads = torch.set_num_threads
torch.set_num_threads = lambda n: _set_num_threads(min(int(n), _cap))

if not os.environ.get("COOPEVA_TRACI") and "--gui" not in sys.argv:
    import libsumo  # noqa: E402
    sys.modules["traci"] = libsumo
    sys.modules["traci.constants"] = libsumo.constants
    backend = f"libsumo {libsumo.__file__}"
else:
    backend = "traci"

MAIN = Path(__file__).resolve().parent / "routing_MARl" / "main.py"
sys.path.insert(0, str(MAIN.parent))

argv = sys.argv[1:]
run_dir = None
if "--output-dir" in argv:
    run_dir = Path(argv[argv.index("--output-dir") + 1]).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    (run_dir / "done.txt").unlink(missing_ok=True)

sys.argv = [str(MAIN)] + argv
print(f"SUMO backend: {backend}; torch threads <= {_cap}; pid {os.getpid()}", flush=True)

status = "ok"
try:
    runpy.run_path(str(MAIN), run_name="__main__")
except SystemExit as e:
    status = f"exit {e.code}"
    raise
except BaseException:
    status = "error\n" + traceback.format_exc()
    raise
finally:
    if run_dir is not None:
        (run_dir / "done.txt").write_text(status, encoding="utf-8")

