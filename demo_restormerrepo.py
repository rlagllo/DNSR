"""
Restormer GPU inference (memory-optimized)
------------------------------------------

목적:
- Restormer 논문의 공개 레포 `demo.py`를 다음 사람이 빠르게 이해하고 유지보수할 수 있도록
  구조화하고 주석을 보강한 버전.

핵심 개선점:
1) 함수 분리: 인자 파싱, 파일 수집, 모델 로딩, 패딩, 타일 추론 등 역할별 함수화
2) 장치/정밀도 자동 선택: CUDA 사용 가능 시 half+autocast, 아니면 float32로 안전 실행
3) 태스크-가중치/하이퍼파라미터 매핑 표준화
4) 입력(단일 파일/폴더) 처리 통일 + 중복 베이스네임 제거
5) 출력 디렉토리 자동 생성 및 프로파일(시간/VRAM) 요약
6) 경계/에러 핸들링: 파일 없음, 타일 인자 검증 등

사용 예시:
python restormer_demo_clean.py \
  --task Real_Denoising \
  --input_dir "/path/to/images" \
  --result_dir "/path/to/out" \
  --tile 720 --tile_overlap 32

주의:
- 사전학습 가중치 경로는 원본 레포 구조를 따릅니다.
- Gaussian_Gray_Denoising 태스크는 1채널 입/출력으로 동작합니다.
"""
from __future__ import annotations

import os
import re
import time
from glob import glob
from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from natsort import natsorted
from runpy import run_path
from tqdm import tqdm
import argparse


# ------------------------------
# I/O 유틸
# ------------------------------
EXTS = ("jpg","JPG","png","PNG","jpeg","JPEG","bmp","BMP")


def list_images(inp_path: str) -> List[str]:
    """단일 파일 또는 폴더에서 이미지 경로 목록을 수집.
    - 단일 파일: 그대로 반환
    - 폴더: 지원 확장자만 수집, 자연 정렬
    - 중복 베이스네임 제거(뒤에 오는 중복은 버림)
    """
    if any(inp_path.endswith(e) for e in EXTS):
        files = [inp_path]
    else:
        files = []
        for e in EXTS:
            files.extend(glob(os.path.join(inp_path, f"*.{e}")))
        files = natsorted(files)

    if not files:
        raise FileNotFoundError(f"No images found in: {inp_path}")

    unique, seen = [], set()
    for f in files:
        b = os.path.splitext(os.path.basename(f))[0]
        if b not in seen:
            seen.add(b)
            unique.append(f)
    return unique


def ensure_out_dir(root: str, task: str) -> str:
    out_dir = os.path.join(root, task)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ------------------------------
# 모델 준비
# ------------------------------
DEFAULT_PARAMETERS: Dict[str, object] = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "dual_pixel_task": False,
}


def task_config(task: str) -> Tuple[str, Dict[str, object]]:
    """태스크에 따른 가중치 경로 및 하이퍼파라미터 덮어쓰기.
    반환: (weights_path, parameters_dict)
    """
    p = dict(DEFAULT_PARAMETERS)

    if task == "Motion_Deblurring":
        w = os.path.join("Motion_Deblurring", "pretrained_models", "motion_deblurring.pth")

    elif task == "Single_Image_Defocus_Deblurring":
        w = os.path.join("Defocus_Deblurring", "pretrained_models", "single_image_defocus_deblurring.pth")

    elif task == "Deraining":
        w = os.path.join("Deraining", "pretrained_models", "deraining.pth")

    elif task == "Real_Denoising":
        w = os.path.join("Denoising", "pretrained_models", "real_denoising.pth")
        p["LayerNorm_type"] = "BiasFree"

    elif task == "Gaussian_Color_Denoising":
        w = os.path.join("Denoising", "pretrained_models", "gaussian_color_denoising_sigma15.pth")
        p["LayerNorm_type"] = "BiasFree"

    elif task == "Gaussian_Gray_Denoising":
        w = os.path.join("Denoising", "pretrained_models", "gaussian_gray_denoising_sigma15.pth")
        p.update({"inp_channels": 1, "out_channels": 1, "LayerNorm_type": "BiasFree"})

    else:
        raise ValueError(f"Unsupported task: {task}")

    return w, p


def load_model(task: str, device: torch.device) -> torch.nn.Module:
    """Restormer 아키텍처 로드 + 가중치 주입.
    - CUDA 사용 시 half precision 파라미터 로드
    """
    weights, params = task_config(task)

    arch = run_path(os.path.join("basicsr", "models", "archs", "restormer_arch.py"))
    model_cls = arch["Restormer"]
    model = model_cls(**params).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("params", ckpt)

    new_state = OrderedDict()
    use_half = device.type == "cuda"
    for k, v in state.items():
        name = k.replace("module.", "")
        if isinstance(v, torch.Tensor) and use_half:
            new_state[name] = v.half()
        else:
            new_state[name] = v

    model.load_state_dict(new_state, strict=True)

    if use_half:
        model = model.half()

    model.eval()
    print_param_report(model)
    return model


def print_param_report(model: torch.nn.Module) -> None:
    tot = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 모델 파라미터 수(개)만 사용. 바이트 환산은 dtype/장치에 따라 달라질 수 있어 참고치만 출력
    print(f"▶ Total params: {tot:,} ({tot/1e6:.2f}M), trainable: {train/1e6:.2f}M")


# ------------------------------
# 전처리/후처리
# ------------------------------

def read_image(fp: str, task: str) -> np.ndarray:
    if task == "Gaussian_Gray_Denoising":
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {fp}")
        return img[..., None]  # (H,W,1)
    else:
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {fp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb


def to_tensor(img: np.ndarray, device: torch.device, use_half: bool) -> torch.Tensor:
    ten = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten.half() if use_half else ten


def unpad_to_shape(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[..., :h, :w]


def to_uint8_image(x: torch.Tensor, task: str) -> np.ndarray:
    x = x.clamp(0, 1)[0]  # (C,H,W)
    arr = (x.permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    if task == "Gaussian_Gray_Denoising":
        return arr  # (H,W,1)
    return arr  # RGB


# ------------------------------
# 패딩/타일 추론
# ------------------------------

def pad_to_multiple(x: torch.Tensor, mult: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    H = ((h + mult - 1) // mult) * mult
    W = ((w + mult - 1) // mult) * mult
    ph, pw = H - h, W - w
    x_p = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x_p, (ph, pw)


def sliding_window_inference(model: torch.nn.Module, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    """타일 기반 추론(블렌딩: 단순 평균 가중).
    - x: (B,C,H,W)
    - tile: 타일 한 변 길이(정사각)
    - overlap: 겹침 크기
    """
    b, c, H, W = x.shape
    assert b == 1, "Batch>1은 현재 지원하지 않습니다."
    if tile <= overlap:
        raise ValueError("tile must be greater than tile_overlap")
    if tile > H or tile > W:
        # 타일이 전체보다 크면 전체 한 번 추론으로 대체
        with torch.no_grad():
            return model(x)

    stride = tile - overlap
    ys = list(range(0, H - tile, stride)) + [H - tile]
    xs = list(range(0, W - tile, stride)) + [W - tile]

    E = torch.zeros_like(x)
    Wm = torch.zeros_like(x)

    for yi in ys:
        for xi in xs:
            patch = x[..., yi : yi + tile, xi : xi + tile]
            with torch.no_grad():
                o = model(patch)
            E[..., yi : yi + tile, xi : xi + tile].add_(o)
            Wm[..., yi : yi + tile, xi : xi + tile].add_(1)

    return E.div_(Wm)


# ------------------------------
# 메인 루프
# ------------------------------

def run_inference(files: List[str], task: str, out_dir: str, tile: int | None, overlap: int, device: torch.device) -> None:
    model = load_model(task, device)
    use_half = device.type == "cuda"

    times, mems = [], []

    for fp in tqdm(files, desc="Running"):
        name = os.path.splitext(os.path.basename(fp))[0]

        # (1) 입력 로드
        img = read_image(fp, task)
        h, w = img.shape[:2]
        inp = to_tensor(img, device, use_half)

        # (2) 패딩
        inp_p, (ph, pw) = pad_to_multiple(inp, mult=8)

        # (3) 프로파일
        if device.type == "cuda":
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        autocast_flag = device.type == "cuda"
        with torch.no_grad():
            if autocast_flag:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    if tile is None:
                        out = model(inp_p)
                    else:
                        out = sliding_window_inference(model, inp_p, tile=tile, overlap=overlap)
            else:
                if tile is None:
                    out = model(inp_p)
                else:
                    out = sliding_window_inference(model, inp_p, tile=tile, overlap=overlap)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

        # (4) 언패드 & 저장
        out = unpad_to_shape(out, h, w)
        res = to_uint8_image(out, task)
        if task == "Gaussian_Gray_Denoising":
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), res)
        else:
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

        print(f"[{name}] {w}x{h}  Time:{t:.3f}s  PeakVRAM:{peak:.1f}MB")
        times.append(t)
        mems.append(peak)

        # 정리
        del inp, inp_p, out, res
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(files)
    avg_t = sum(times) / n if n else 0
    avg_m = sum(mems) / n if n else 0
    print(f"\nSummary: {n} imgs, AvgTime:{avg_t:.3f}s, AvgVRAM:{avg_m:.1f}MB")


# ------------------------------
# Argparse & 엔트리포인트
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restormer GPU inference (memory-optimized)")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory or single image path")
    parser.add_argument("--result_dir", required=True, type=str, help="Directory for restored results")
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=[
            "Motion_Deblurring",
            "Single_Image_Defocus_Deblurring",
            "Deraining",
            "Real_Denoising",
            "Gaussian_Gray_Denoising",
            "Gaussian_Color_Denoising",
        ],
        help="Task to run",
    )
    parser.add_argument("--tile", type=int, default=None, help="Square tile size (e.g. 720)")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Tile overlap (pixels)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 장치/정밀도 결정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("▶ Using CUDA (fp16 autocast)")
    else:
        print("▶ Using CPU (fp32)")

    files = list_images(args.input_dir)
    out_dir = ensure_out_dir(args.result_dir, args.task)

    # 타일 옵션 검증(선택 사항)
    if args.tile is not None and args.tile <= 0:
        raise ValueError("--tile must be a positive integer or omitted")
    if args.tile is not None and args.tile_overlap < 0:
        raise ValueError("--tile_overlap must be >= 0")

    run_inference(
        files=files,
        task=args.task,
        out_dir=out_dir,
        tile=args.tile,
        overlap=args.tile_overlap,
        device=device,
    )


if __name__ == "__main__":
    main()
