"""Run integration checks and record a small, reproducible verification report."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
temporary = ROOT / ".runtime" / "test-temp"
temporary.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(temporary)
os.environ["TMP"] = str(temporary)

import nbformat


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    notebooks = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = sum(output.output_type == "error" for cell in cells for output in cell.outputs)
        executed = all(cell.execution_count is not None for cell in cells)
        notebooks.append({"file": path.name, "code_cells": len(cells), "executed": executed, "errors": errors})
    notebook_pass = all(item["executed"] and not item["errors"] for item in notebooks)
    report = {
        "tests_run": result.testsRun,
        "test_failures": len(result.failures),
        "test_errors": len(result.errors),
        "notebooks": notebooks,
        "status": "passed" if result.wasSuccessful() and notebook_pass else "failed",
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "streamlit", "matplotlib", "opencv-python", "nbclient"]
        },
    }
    (ROOT / "reports/verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verification: {report['status']}. {len(notebooks)} executed notebooks checked.")
    raise SystemExit(0 if report["status"] == "passed" else 1)
