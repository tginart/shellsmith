#!/usr/bin/env python3
"""Stage verified supervision templates for a fresh run; never launch or resume."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shellsmith.jobs.storage import write


def stage(root: Path) -> dict:
    root = root.absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Run controls refuse symlinked destinations")
    if any((root / name).exists() for name in ("launch.json", "dispatch-intent.json", "launch-dispatch.json", "owner-limit.json")):
        raise ValueError("Never replace controls of a dispatched or historical run")
    if (root / "instance.json").exists():
        instance = json.loads((root / "instance.json").read_text())
        if instance.get("status") not in ("prepared", "staged"):
            raise ValueError("Only a fresh, prepared run may receive new controls")
    receipt_path = root / "control-template.json"
    if receipt_path.is_symlink():
        raise ValueError("Refusing a symlinked control receipt")
    scripts = Path(__file__).resolve().parent
    contents = {name: (scripts / source).read_bytes() for name, source in {
        "workflow.py": "studio_micro_workflow.py", "supervisor.py": "studio_micro_supervisor.py",
        "native_preflight.py": "studio_micro_preflight.py",
    }.items()}
    for name, content in contents.items():
        destination = root / name
        if destination.is_symlink() or (destination.exists() and destination.read_bytes() != content):
            raise ValueError(f"Refusing to replace different existing controls: {name}")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, content in contents.items():
        destination = root / name
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(content)
            destination.chmod(0o600)
    receipt = {"status": "staged_not_launched", "files": {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
        "required_runtime_module": "shellsmith.jobs.studio_monitor",
        "deadline_changed": False, "historical_evidence_changed": False}
    write(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().run_root)))
