#!/usr/bin/env bash
# MotLab installer — sets up MotrixSim + workspace packages via uv, conda or
# plain pip. Safe to re-run.
#
#   bash scripts/install.sh                        # auto-detect + rsl_rl extra
#   bash scripts/install.sh --method uv
#   bash scripts/install.sh --method conda --rllib skrl
#   bash scripts/install.sh --method pip   --rllib both
#   bash scripts/install.sh --method pip   --python 3.11
#   bash scripts/install.sh --no-extras            # envs only, no RL framework
#   bash scripts/install.sh --dev                  # + dev tooling (pytest, ruff)

set -euo pipefail

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
METHOD="auto"                # auto | uv | conda | pip
RLLIB="rslrl"                # rslrl | none
PYTHON_VERSION="3.10"
CONDA_ENV_NAME="motlab"
VENV_DIR=".venv"
DEV=0
VERIFY=1

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options
  --method {auto,uv,conda,pip}   Installer to use (default: auto).
  --rllib  {rslrl,skrl,both,none} RL framework extra to install (default: rslrl).
  --python <ver>                 Python version for new venv/conda env (default: 3.10).
  --env <name>                   Conda env name (default: motlab).
  --venv-dir <path>              Path for pip/uv venv (default: .venv).
  --no-extras                    Skip the RL framework extra (envs only).
  --dev                          Also install dev tooling (pytest, ruff, pre-commit).
  --no-verify                    Skip the post-install import check.
  -h | --help                    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --rllib) RLLIB="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --env) CONDA_ENV_NAME="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --no-extras) RLLIB="none"; shift ;;
        --dev) DEV=1; shift ;;
        --no-verify) VERIFY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_red=$'\033[31m'; c_reset=$'\033[0m'
info()  { printf "%s==>%s %s\n" "$c_green"  "$c_reset" "$*"; }
warn()  { printf "%s!! %s%s\n" "$c_yellow" "$*" "$c_reset"; }
die()   { printf "%sxx %s%s\n" "$c_red"    "$*" "$c_reset" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Locate the motlab root (script is scripts/install.sh)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
info "MotLab root: $ROOT_DIR"

# ---------------------------------------------------------------------------
# Resolve auto method
# ---------------------------------------------------------------------------
if [[ "$METHOD" == "auto" ]]; then
    if command -v uv >/dev/null 2>&1; then METHOD="uv"
    elif command -v conda >/dev/null 2>&1; then METHOD="conda"
    else METHOD="pip"; fi
    info "Auto-selected installer: $METHOD"
fi

# ---------------------------------------------------------------------------
# Build extras string: e.g. "rslrl" -> "packages/motlab_rl[rslrl]"
# ---------------------------------------------------------------------------
case "$RLLIB" in
    rslrl|skrl)
        RL_PKG_SPEC="packages/motlab_rl[$RLLIB]" ;;
    both)
        RL_PKG_SPEC="packages/motlab_rl[rslrl,skrl]" ;;
    none)
        RL_PKG_SPEC="packages/motlab_rl" ;;
    *)
        die "Unknown --rllib value: $RLLIB (expected rslrl|skrl|both|none)" ;;
esac

# ---------------------------------------------------------------------------
# Install by method
# ---------------------------------------------------------------------------
install_uv() {
    info "Using uv"
    command -v uv >/dev/null 2>&1 || die "uv not found. Install: https://docs.astral.sh/uv/"

    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    local extra=()
    case "$RLLIB" in
        rslrl|skrl) extra+=(--extra "$RLLIB") ;;
        both)       extra+=(--extra rslrl --extra skrl) ;;
        none)       : ;;
    esac
    [[ "$DEV" == "1" ]] && extra+=(--extra dev)

    # Sync all workspace packages (motlab core + assets + tasks + rl).
    uv sync --all-packages "${extra[@]}"

    PYTHON_BIN="$VENV_DIR/bin/python"
    info "uv sync complete. Activate with: source $VENV_DIR/bin/activate"
}

install_conda() {
    info "Using conda (env: $CONDA_ENV_NAME)"
    command -v conda >/dev/null 2>&1 || die "conda not found."

    # Source conda.sh so `conda activate` works in this subshell.
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        info "Reusing existing conda env '$CONDA_ENV_NAME'"
    else
        info "Creating conda env '$CONDA_ENV_NAME' from environment.yml"
        conda env create -f environment.yml -n "$CONDA_ENV_NAME"
    fi

    conda activate "$CONDA_ENV_NAME"

    # Editable installs for workspace packages.
    pip install -e packages/motlab -e packages/motlab_assets -e packages/motlab_tasks
    pip install -e "$RL_PKG_SPEC"

    [[ "$DEV" == "1" ]] && pip install pytest ruff pre-commit

    PYTHON_BIN="$(command -v python)"
    info "conda env ready. Activate with: conda activate $CONDA_ENV_NAME"
}

install_pip() {
    info "Using plain pip (venv: $VENV_DIR)"
    local py_bin="python${PYTHON_VERSION}"
    command -v "$py_bin" >/dev/null 2>&1 || die "python${PYTHON_VERSION} not found. Install it or pick another --python."

    if [[ ! -d "$VENV_DIR" ]]; then
        if ! "$py_bin" -m venv "$VENV_DIR" 2>/dev/null; then
            warn "venv creation failed — do you have python${PYTHON_VERSION}-venv installed?"
            die "On Debian/Ubuntu: sudo apt install python${PYTHON_VERSION}-venv"
        fi
    else
        info "Reusing existing venv at $VENV_DIR"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install motrixsim
    "$PYTHON_BIN" -m pip install -e packages/motlab -e packages/motlab_assets -e packages/motlab_tasks
    "$PYTHON_BIN" -m pip install -e "$RL_PKG_SPEC"

    [[ "$DEV" == "1" ]] && "$PYTHON_BIN" -m pip install pytest ruff pre-commit

    info "venv ready. Activate with: source $VENV_DIR/bin/activate"
}

case "$METHOD" in
    uv)    install_uv ;;
    conda) install_conda ;;
    pip)   install_pip ;;
    *) die "Unknown --method '$METHOD'" ;;
esac

# ---------------------------------------------------------------------------
# Post-install verification
# ---------------------------------------------------------------------------
if [[ "$VERIFY" == "1" ]]; then
    info "Verifying install (import motlab + motlab_tasks + step cartpole once) …"
    "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch

    import motlab
    import motlab_tasks  # registers built-in envs

    assert "cartpole" in motlab.list_envs(), motlab.list_envs()
    cfg = motlab.make_cfg("cartpole")
    cfg.scene.num_envs = 2
    env = motlab.ManagerBasedRLEnv(cfg, device="cpu")
    env.reset()
    actions = torch.zeros(env.num_envs, env.action_dim, dtype=torch.float32, device=env.device)
    env.step(actions)
    print("OK — motlab + motlab_tasks imported, cartpole stepped.")
except Exception as exc:
    print(f"FAILED: {exc}", file=sys.stderr)
    raise
PY
    info "✔ verified"
fi

info "Done."
case "$METHOD" in
    uv)    echo "   Next:  source $VENV_DIR/bin/activate && python scripts/view.py --env cartpole" ;;
    conda) echo "   Next:  conda activate $CONDA_ENV_NAME && python scripts/view.py --env cartpole" ;;
    pip)   echo "   Next:  source $VENV_DIR/bin/activate && python scripts/view.py --env cartpole" ;;
esac
