# Shellsmith sandbox image: one container per episode (vat + Game Master).
#
# Pi supplies the Player harness SDK; Codex is the container Game Master CLI.
# Pi GM execution is disabled because its unconfined tools expose credentials.
# Install its complete npm package, not a host ``dist/`` directory: the compiled
# files import runtime dependencies from node_modules. Pin both agent CLIs so a
# rebuild cannot silently change rollout behavior.

FROM node:22.22.0-bookworm-slim

# Optional overrides; defaults are read from the same config as build.sh.
ARG PI_CODING_AGENT_VERSION
ARG CODEX_VERSION
COPY src/shellsmith/config.py /opt/shellsmith-build-config.py

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/shellsmith/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        tini \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN PI_CODING_AGENT_VERSION="${PI_CODING_AGENT_VERSION:-$(python3 -c 'import runpy; print(runpy.run_path("/opt/shellsmith-build-config.py")["PI_CODING_AGENT_VERSION"])')}" \
    && CODEX_VERSION="${CODEX_VERSION:-$(python3 -c 'import runpy; print(runpy.run_path("/opt/shellsmith-build-config.py")["CODEX_CLI_VERSION"])')}" \
    && npm install -g --no-audit --no-fund \
        "@earendil-works/pi-coding-agent@${PI_CODING_AGENT_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
    && pi --version \
    && codex --version

# The core package is stdlib-only, so PYTHONPATH is enough (no pip install).
# World files are NOT copied: the Player shares this container, so the host
# uploads exactly one private world file per episode, outside the vat workspace.
COPY src /opt/shellsmith/src

ENV SHELLSMITH_PI_SDK_MODULE_PATH=/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/dist/index.js \
    SHELLSMITH_PI_BIN=/usr/local/bin/pi

RUN test -f "${SHELLSMITH_PI_SDK_MODULE_PATH}" \
    && node -e "import(process.env.SHELLSMITH_PI_SDK_MODULE_PATH)"

RUN mkdir -p /workspace /opt/shellsmith /logs/artifacts /logs/in-progress
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
