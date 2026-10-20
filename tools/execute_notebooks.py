"""Execute every notebook with this interpreter and preserve outputs."""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for variable, folder in {
    "JUPYTER_RUNTIME_DIR": ".runtime/jupyter",
    "IPYTHONDIR": ".runtime/ipython",
    "MPLCONFIGDIR": ".runtime/matplotlib",
}.items():
    destination = ROOT / folder
    destination.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(destination)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


if __name__ == "__main__":
    reports = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(
            notebook, km=manager, timeout=240, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        count = sum(cell.cell_type == "code" for cell in notebook.cells)
        seconds = round(time.monotonic() - started, 2)
        reports.append({"notebook": path.name, "status": "passed", "code_cells": count, "seconds": seconds})
        print(f"PASS {path.name}: {count} code cells in {seconds}s", flush=True)
    (ROOT / "reports/notebook_execution.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
