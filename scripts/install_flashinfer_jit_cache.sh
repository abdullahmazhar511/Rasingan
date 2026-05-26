#!/usr/bin/env bash
# Install flashinfer-jit-cache into the vllmenv venv to bypass the GDN
# JIT-compile hang that wedges the supervisor vLLM on Qwen3.5 hybrid models.
# Refs: https://github.com/vllm-project/vllm/issues/35496
#       https://github.com/vllm-project/vllm/issues/37250
#
# The flashinfer.ai/whl index is just an HTML page; the wheels themselves
# live on GitHub Releases. We resolve the GitHub URL from the index and
# download from there directly (CDN-backed, much faster).

set -euo pipefail

VENV="${VENV:-/home/asbahk/vllmenv}"
PY="$VENV/bin/python"
INDEX_URL="${INDEX_URL:-https://flashinfer.ai/whl/cu129/flashinfer-jit-cache/}"

# Pick an installer: prefer venv's own pip, fall back to `uv pip` (uv-created
# venvs ship without pip), then to `python -m pip` if that module is bundled.
if [ -x "$VENV/bin/pip" ]; then
    INSTALL_CMD=("$VENV/bin/pip" install --no-deps)
elif command -v uv >/dev/null 2>&1; then
    INSTALL_CMD=(uv pip install --no-deps --python "$PY")
elif "$PY" -c "import pip" >/dev/null 2>&1; then
    INSTALL_CMD=("$PY" -m pip install --no-deps)
else
    echo "ERROR: no installer available — venv has no pip and uv is not on PATH" >&2
    echo "Fix: 'pip install uv' in your shell, or add pip to the venv with:" >&2
    echo "     $PY -m ensurepip --upgrade" >&2
    exit 1
fi
echo "[i]   installer: ${INSTALL_CMD[*]}"
ARCH_TAG="${ARCH_TAG:-cp39-abi3-manylinux_2_28_x86_64}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp}"

if [ ! -x "$PY" ]; then
    echo "ERROR: python not found at $PY" >&2
    exit 1
fi

echo "[1/5] Detecting installed flashinfer version in $VENV"
FI_VERSION="$("$PY" -c 'import flashinfer; print(flashinfer.__version__)')"
echo "      flashinfer == $FI_VERSION"

if "$PY" -c "import flashinfer_jit_cache" >/dev/null 2>&1; then
    echo "[!]   flashinfer_jit_cache already installed — nothing to do."
    "$PY" -c "import flashinfer, flashinfer_jit_cache; print('flashinfer', flashinfer.__version__, 'jit_cache OK')"
    exit 0
fi

WHEEL="flashinfer_jit_cache-${FI_VERSION}+cu129-${ARCH_TAG}.whl"

echo "[2/5] Resolving GitHub Releases URL from the flashinfer index"
INDEX_HTML="$(curl -sS "$INDEX_URL")"
# Each entry in the index looks like:
#   <a href="https://github.com/.../v0.6.8.post1/flashinfer_jit_cache-0.6.8.post1+cu129-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=...">...</a>
# Extract the href whose URL ends with our exact wheel filename.
RESOLVED_URL="$(printf '%s\n' "$INDEX_HTML" \
    | grep -oE 'https://github\.com/[^"#]+\.whl' \
    | grep -F "/${WHEEL}" \
    | head -1)"

if [ -z "$RESOLVED_URL" ]; then
    echo "ERROR: could not find $WHEEL in $INDEX_URL" >&2
    echo "Available wheels for ${ARCH_TAG}:" >&2
    printf '%s\n' "$INDEX_HTML" \
        | grep -oE "flashinfer_jit_cache-[^\"]+${ARCH_TAG}\\.whl" \
        | sort -u >&2
    exit 1
fi

DEST="${DOWNLOAD_DIR}/${WHEEL}"
echo "      wheel : $WHEEL"
echo "      url   : $RESOLVED_URL"
echo "      dest  : $DEST"

echo "[3/5] Downloading (resumable, GitHub CDN — expect ~3-6 GB)"
# wget --continue picks up where it left off if the download is interrupted
wget --continue --show-progress -O "$DEST" "$RESOLVED_URL"
ls -lh "$DEST"

echo "[4/5] Installing into $VENV"
"${INSTALL_CMD[@]}" "$DEST"

echo "[5/5] Verification"
"$PY" -c "
import flashinfer, flashinfer_jit_cache, os, sys
print('python              :', sys.version.split()[0])
print('flashinfer          :', flashinfer.__version__)
print('flashinfer_jit_cache: OK (path=' + os.path.dirname(flashinfer_jit_cache.__file__) + ')')
"
echo
echo "[done] flashinfer-jit-cache installed. Next: launch the supervisor with"
echo "       SUPERVISOR_PYTHON=$PY in your run script."