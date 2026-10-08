"""Recalculate saved metrics and validate patient-group separation locally."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acs_research.paths import study_paths
from acs_research.validation import validate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["reference", "current"], default="reference")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    paths = study_paths(ROOT)
    directory = paths.reference if args.run == "reference" else paths.run
    audit = validate_run(directory, check_model=not args.skip_model)
    (directory / "repository_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
