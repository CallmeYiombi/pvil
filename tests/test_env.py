"""가벼운 환경 검사. 무거운 점검(모델 로드, 벤치마크)은 scripts/check_env.py에서 한다."""

import os
from pathlib import Path

import pytest

HOME = Path(os.path.expanduser("~")).resolve()
PATH_VARS = ["HF_HOME", "TORCH_HOME", "CELLPOSE_LOCAL_MODELS_PATH", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
             "UV_PYTHON_BIN_DIR", "UV_PROJECT_ENVIRONMENT", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
             "TORCHINDUCTOR_CACHE_DIR", "MPLCONFIGDIR", "PIP_CACHE_DIR"]


@pytest.mark.parametrize("var", PATH_VARS)
def test_cache_paths_outside_home(var):
    value = os.environ.get(var)
    assert value, f"{var} 미설정: source scripts/env.sh 필요"
    assert not Path(value).resolve().is_relative_to(HOME), f"{var}={value} 가 ~ 아래에 있음"


def test_imports():
    import cellpose, copairs, cp_measure, harmonypy, peft, pycytominer, rdkit, scanpy, scib_metrics, timm, transformers  # noqa: F401


def test_cuda_b200():
    import torch

    assert torch.cuda.is_available()
    assert "B200" in torch.cuda.get_device_name(0)


def test_run_info_git_hash():
    from pvil.eval import run_info

    info = run_info.collect(config_path="configs/base_cpbg.yaml", seed=0)
    assert len(info["git_hash"].split("-")[0]) == 40
    assert info["packages"]["torch"] is not None
