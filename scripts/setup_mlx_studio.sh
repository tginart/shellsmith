#!/bin/bash
# Dedicated user-owned tools only. Run explicitly on the selected Studio.
set -euo pipefail
umask 077
shellsmith_checkout="$(cd "$(dirname "$0")/.." && pwd)"
shellsmith_prefix="${1:-$HOME/.local/share/shellsmith/studio-v1}"
shellsmith_conda="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
shellsmith_source_env="${2:-}"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This setup requires an Apple Silicon Mac.' >&2; exit 1
fi
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 26 ]]; then
  echo 'The pinned MLX wheels require macOS 26 or later.' >&2; exit 1
fi
if [[ -L "$shellsmith_prefix" ]]; then echo 'Refusing a symlinked installation.' >&2; exit 1; fi
mkdir -p "$shellsmith_prefix"
if [[ -e "$shellsmith_prefix/env" && ! -f "$shellsmith_prefix/.shellsmith-owned" ]]; then
  echo 'An unmanaged environment exists; choose a new prefix.' >&2; exit 1
fi
touch "$shellsmith_prefix/.shellsmith-owned"
if [[ ! -x "$shellsmith_prefix/env/bin/python" ]]; then
  if [[ -n "$shellsmith_source_env" ]]; then
    # Cloning changes only the new prefix, never the source environment.
    "$shellsmith_conda" create --yes --offline --override-channels -c conda-forge --prefix "$shellsmith_prefix/env" --clone "$shellsmith_source_env"
  else
    "$shellsmith_conda" create --yes --override-channels -c conda-forge --prefix "$shellsmith_prefix/env" python=3.12 pip
  fi
fi
"$shellsmith_prefix/env/bin/python" -m pip install --require-hashes -r "$shellsmith_checkout/src/shellsmith/mlx_requirements.lock"
# Telemetry shares the locked Pydantic runtime. Install the SDK and stats
# collector without re-resolving or upgrading the numerical environment.
"$shellsmith_prefix/env/bin/python" -m pip install --no-deps --require-hashes -r "$shellsmith_checkout/src/shellsmith/telemetry_requirements.lock"
# pip cannot hash a VCS checkout. pyproject.toml pins a full commit SHA, which
# is content-addressed, and stays the single source for the SDK version.
shellsmith_backprop="$("$shellsmith_prefix/env/bin/python" - "$shellsmith_checkout/pyproject.toml" <<'PY'
import re, sys, tomllib
dependencies = tomllib.load(open(sys.argv[1], 'rb'))['project']['dependencies']
pin = next(item for item in dependencies if item.startswith('backprop-foundry'))
if not re.search(r'\.git@[0-9a-f]{40}$', pin):
    raise SystemExit('backprop-foundry must pin a full commit SHA')
print(pin)
PY
)"
"$shellsmith_prefix/env/bin/python" -m pip install --no-deps "$shellsmith_backprop"
if [[ ! -x "$shellsmith_prefix/node/bin/node" ]]; then
  shellsmith_download="$(mktemp -d "$shellsmith_prefix/.node-download.XXXXXX")"
  trap 'rm -rf "$shellsmith_download"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 https://nodejs.org/dist/v22.22.0/node-v22.22.0-darwin-arm64.tar.gz --output "$shellsmith_download/node.tar.gz"
  curl --fail --location --proto '=https' --tlsv1.2 https://nodejs.org/dist/v22.22.0/SHASUMS256.txt --output "$shellsmith_download/SHASUMS256.txt"
  "$shellsmith_prefix/env/bin/python" - "$shellsmith_download" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = [line.split() for line in (root/'SHASUMS256.txt').read_text().splitlines()]
wanted = next(sha for sha, name in rows if name == 'node-v22.22.0-darwin-arm64.tar.gz')
if hashlib.sha256((root/'node.tar.gz').read_bytes()).hexdigest() != wanted:
    raise SystemExit('Node artifact hash mismatch')
PY
  mkdir "$shellsmith_prefix/node"
  tar -xzf "$shellsmith_download/node.tar.gz" --strip-components=1 -C "$shellsmith_prefix/node"
fi
export PATH="$shellsmith_prefix/env/bin:$shellsmith_prefix/node/bin:$shellsmith_prefix/tools/node_modules/.bin:$PATH"
export MLX_ENABLE_TF32=0
if [[ "$(node --version)" != v22.22.0 ]]; then echo 'Node identity mismatch.' >&2; exit 1; fi
npm install --prefix "$shellsmith_prefix/tools" --save-exact --ignore-scripts --no-audit --no-fund @openai/codex@0.159.2 @earendil-works/pi-coding-agent@0.84.1
python -m pip install --no-deps -e "$shellsmith_checkout"
# --no-deps skips dependency checks, so prove the default telemetry SDK loads.
python -c 'import backprop, psutil'
(cd "$shellsmith_checkout/src/shellsmith/episode/vat_runner" && npm ci --ignore-scripts --no-audit --no-fund)
python -c 'import json; from shellsmith.jobs.readiness import probe; p=probe(); print(json.dumps(p)); raise SystemExit(0 if p["ready"] else 1)'
echo 'Authenticate explicitly with: codex login --device-auth'
echo 'Setup does not download a model or stop an existing inference service.'
