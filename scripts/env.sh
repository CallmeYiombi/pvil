# PVIL 환경 변수. 사용: source scripts/env.sh
# ~/(/home/jovyan)에 아무것도 쓰지 않도록 모든 캐시·venv 경로를 /home/genetics/yiombi 아래로 돌린다.

export PVIL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PVIL_DATA=/home/genetics/yiombi/pvil_data
export PVIL_CACHE=/home/genetics/yiombi/hf_cache

# 모델 가중치 (공용 캐시 재사용)
export HF_HOME="$PVIL_CACHE"
export HF_HUB_CACHE="$PVIL_CACHE/hub"
export TORCH_HOME="$PVIL_CACHE/torch"
export CELLPOSE_LOCAL_MODELS_PATH="$PVIL_CACHE/cellpose"

# 패키지·컴파일 캐시
export PIP_CACHE_DIR="$PVIL_CACHE/pip"
export UV_CACHE_DIR="$PVIL_CACHE/uv"
export UV_PYTHON_INSTALL_DIR="$PVIL_CACHE/uv_python"
export UV_PYTHON_BIN_DIR="$PVIL_CACHE/uv_python/bin"
export UV_PYTHON_PREFERENCE=only-managed
export UV_TOOL_DIR="$PVIL_CACHE/uv_tools"
export UV_TOOL_BIN_DIR="$PVIL_CACHE/uv_tools/bin"
export UV_PROJECT_ENVIRONMENT=/home/genetics/yiombi/venv_pvil
export XDG_CACHE_HOME="$PVIL_CACHE/xdg"
export TRITON_CACHE_DIR="$PVIL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$PVIL_CACHE/inductor"
export MPLCONFIGDIR="$PVIL_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$PVIL_CACHE/numba"
# scib-metrics(JAX)는 CPU에서 돌려 torch CUDA와 충돌하지 않게 한다
export JAX_PLATFORMS=cpu

mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

# 공유 서버: 우리 작업은 GPU 1, 2만 쓴다(사용자 지시 2026-10-14). 스크립트가 따로 지정하면 그 값을 따른다
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
