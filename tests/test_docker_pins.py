"""Drive the shipped build wrapper and Docker RUN shell with captured external tools.

Docker itself is an external dependency: these tests verify the actual command
assembly and source-context boundary without claiming an image was built.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from shellsmith.paths import PROJECT_ROOT


def _configuration(path: Path) -> None:
    source = (PROJECT_ROOT / "src/shellsmith/config.py").read_text()
    import re
    source = re.sub(r'^PI_CODING_AGENT_VERSION = .*$', 'PI_CODING_AGENT_VERSION = "9.8.7"', source, flags=re.M)
    source = re.sub(r'^CODEX_CLI_VERSION = .*$', 'CODEX_CLI_VERSION = "6.5.4"', source, flags=re.M)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _tool(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("overrides", [False, True])
def test_real_build_wrapper_reads_pins_from_committed_config(tmp_path: Path, overrides: bool) -> None:
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    for name in ("build.sh", "Dockerfile"):
        shutil.copyfile(PROJECT_ROOT / "docker" / name, repo / "docker" / name)
    _configuration(repo / "src/shellsmith/config.py")
    (repo / "worlds").mkdir()
    (repo / "worlds/private.md").write_text("must never enter build context")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "src", "docker"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Shellsmith Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "fixture"], check=True)
    tools = tmp_path / "bin"
    tools.mkdir()
    capture = tmp_path / "docker-args.json"
    _tool(tools / "docker", "import json, os, pathlib, sys\n"
          "context = pathlib.Path(sys.argv[-1])\n"
          "assert (context / 'src/shellsmith/config.py').is_file()\n"
          "assert not (context / 'worlds').exists()\n"
          "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
    temp = tmp_path / "temp"
    temp.mkdir()
    env = {key: value for key, value in os.environ.items() if key not in {"PI_CODING_AGENT_VERSION", "CODEX_VERSION", "DOCKER_PLATFORM", "IMAGE_TAG"}}
    env.update(PATH=str(tools) + os.pathsep + env["PATH"], CAPTURE=str(capture), TMPDIR=str(temp))
    if overrides:
        env.update(PI_CODING_AGENT_VERSION="1.2.3", CODEX_VERSION="4.5.6")
    result = subprocess.run(["bash", str(repo / "docker/build.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    values = [args[index + 1] for index, value in enumerate(args) if value == "--build-arg"]
    assert values == (["PI_CODING_AGENT_VERSION=1.2.3", "CODEX_VERSION=4.5.6"] if overrides else ["PI_CODING_AGENT_VERSION=9.8.7", "CODEX_VERSION=6.5.4"])
    assert not Path(args[-1]).exists(), "wrapper must clean its private source context"


def test_dockerfile_default_install_stage_reads_the_same_config(tmp_path: Path) -> None:
    dockerfile = (PROJECT_ROOT / "docker/Dockerfile").read_text()
    assert "ARG PI_CODING_AGENT_VERSION=" not in dockerfile
    assert "ARG CODEX_VERSION=" not in dockerfile
    assert "COPY src/shellsmith/config.py /opt/shellsmith-build-config.py" in dockerfile
    # Execute the production RUN instruction unchanged except its container
    # config mount point. Only the external installers/CLIs are captured.
    stage = next(instruction for instruction in dockerfile.replace("\\\n", "").splitlines() if instruction.startswith("RUN PI_CODING_AGENT_VERSION="))[4:]
    config = tmp_path / "config.py"
    _configuration(config)
    stage = stage.replace("/opt/shellsmith-build-config.py", str(config))
    tools = tmp_path / "bin"
    tools.mkdir()
    capture = tmp_path / "npm-args.json"
    _tool(tools / "npm", "import json, os, pathlib, sys\npathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
    for name in ("pi", "codex"):
        _tool(tools / name, "print('captured version command')\n")
    env = {key: value for key, value in os.environ.items() if key not in {"PI_CODING_AGENT_VERSION", "CODEX_VERSION"}}
    env.update(PATH=str(tools) + os.pathsep + env["PATH"], CAPTURE=str(capture))
    result = subprocess.run(["sh", "-c", stage], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert "@earendil-works/pi-coding-agent@9.8.7" in args
    assert "@openai/codex@6.5.4" in args
