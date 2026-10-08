"""Execute ACS ML notebooks in order; all outputs stay in ignored private storage."""
from pathlib import Path
import argparse
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acs_research.paths import study_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["01", "02", "03", "all"], default="01")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    paths = study_paths(ROOT)
    if not (paths.inputs / "curated_clinical.csv").exists() or not (paths.inputs / "source_measurements.csv").exists():
        raise FileNotFoundError("Для пересчёта нужны локальные CSV в studies/acs/data/private/inputs. См. data/README.md.")
    paths.run.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(paths.run / ".matplotlib"))
    notebooks = sorted((paths.study / "machine_learning/notebooks").glob("*.ipynb"))
    for path in notebooks:
        if args.step != "all" and not path.name.startswith(args.step):
            continue
        notebook = nbformat.read(path, as_version=4)
        print("Executing", path.name, flush=True)
        client = NotebookClient(notebook, kernel_name=args.kernel, timeout=None,
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, paths.run / path.name)
        print("Saved locally:", (paths.run / path.name).relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
