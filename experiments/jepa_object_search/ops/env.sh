# Source before launching project jobs to keep caches on the data volume.
export NAV_ROOT=/root/autodl-tmp/jepa_object_search
export HF_HOME="$NAV_ROOT/cache/huggingface"
export TORCH_HOME="$NAV_ROOT/cache/torch"
export PIP_CACHE_DIR="$NAV_ROOT/cache/pip"
export UV_CACHE_DIR="$NAV_ROOT/cache/uv"
export XDG_CACHE_HOME="$NAV_ROOT/cache/xdg"
export TMPDIR="$NAV_ROOT/cache/tmp"
export CONDA_PKGS_DIRS="$NAV_ROOT/cache/conda-pkgs"
export CONDA_ENVS_PATH="$NAV_ROOT/envs"
export TRITON_CACHE_DIR="$NAV_ROOT/cache/triton"
export CUDA_CACHE_PATH="$NAV_ROOT/cache/cuda"
export PYTHONPYCACHEPREFIX="$NAV_ROOT/cache/pycache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$CONDA_PKGS_DIRS" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$PYTHONPYCACHEPREFIX"
