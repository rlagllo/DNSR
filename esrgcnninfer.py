"""
ESRGCNN X-ray Inference (Clean, Well-Commented)
------------------------------------------------
- 두 입력 모드 지원:
  1) LR 모드: 입력이 이미 저해상도(LR) → 모델 x4 업스케일 후 저장
  2) HR 모드: 입력이 고해상도(HR) → 1/4 다운(BICUBIC)하여 LR 생성 → 모델 x4 복원 → HR과 정렬된 결과 저장
- 대형 이미지 메모리 절약: 재귀적 chop-forward + 오버랩 블렌딩(평균)
- 패딩: 타일 경계/짝수 분할 안정성 위해 reflect pad
- 디바이스/정밀도: CUDA 사용 시 autocast(fp16) 자동 적용

예시 실행
---------
# 입력이 이미 LR인 경우(x4 업스케일)
python esrgcnn_xray_infer_clean.py \
  --model esrgcnn --ckpt_path checkpoint/esrgcnn.pth \
  --test_data_dir ./xraydata/LR \
  --sample_dir ./results \
  --scale 4 --input_mode lr --tile_min_size 160000 --shave 20

# HR 기준 벤치마크(다운1/4→업4)
python esrgcnn_xray_infer_clean.py \
  --model esrgcnn --ckpt_path checkpoint/esrgcnn.pth \
  --test_data_dir ./xraydata/HR \
  --sample_dir ./results \
  --scale 4 --input_mode hr --tile_min_size 160000 --shave 20

필요 모듈: ESRGCNNWrapper, xraydataset(TestDataset 대체 가능), model/<name>.py 내 Net
"""
from __future__ import annotations

import os
import glob
import json
import time
import argparse
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from wrap import ESRGCNNWrapper
from dataset import xraydataset  # (또는 TestDataset)


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESRGCNN X-ray Inference (clean)")
    p.add_argument("--model", required=True, type=str, help="model/<name>.py 의 모듈명")
    p.add_argument("--ckpt_path", required=True, type=str, help="학습 가중치 .pth 경로")
    p.add_argument("--test_data_dir", required=True, type=str, help="입력 이미지 폴더")
    p.add_argument("--sample_dir", required=True, type=str, help="결과 저장 루트 폴더")
    p.add_argument("--scale", type=int, default=4, help="업스케일 배수 (기본 4)")
    p.add_argument("--group", type=int, default=1, help="모델 group 설정")
    p.add_argument("--input_mode", type=str, default="lr", choices=["lr","hr"],
                   help="lr: 입력이 저해상도 / hr: 입력이 고해상도(1/4 다운 후 복원)")
    p.add_argument("--shave", type=int, default=20, help="타일 오버랩 크기")
    p.add_argument("--tile_min_size", type=int, default=160000, help="타일 최소 픽셀 수 (재귀 분할 기준)")
    return p.parse_args()


# ------------------------------
# I/O 유틸
# ------------------------------

EXTS = ("*.png","*.jpg","*.jpeg","*.bmp","*.PNG","*.JPG","*.JPEG","*.BMP")

def list_images(d: str):
    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(d, e))
    return sorted(files)


def save_tensor_image(x: torch.Tensor, path: str, rgb: bool = True) -> None:
    """x: (C,H,W), float[0,1]"""
    x = x.clamp(0,1).detach().cpu().permute(1,2,0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    if rgb:
        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:
        # 단일채널 지원 필요 시 확장
        cv2.imwrite(path, x)


# ------------------------------
# 패딩/타일 추론
# ------------------------------

def reflect_pad_to_multiple(x: torch.Tensor, mult: int = 1) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """(1,C,H,W) 텐서를 mult 배수로 맞춰 reflect pad"""
    b, c, h, w = x.shape
    H = ((h + mult - 1) // mult) * mult
    W = ((w + mult - 1) // mult) * mult
    ph, pw = H - h, W - w
    x_p = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x_p, (ph, pw)


def forward_chop_recursive(net, lr, scale: int, shave: int = 20, min_size: int = 160000, device: str = "cuda"):
    """재귀적 chop-forward. lr: (1,C,H,W), net(lr, scale) -> (1,C,scale*H, scale*W) 가정"""
    _, _, h, w = lr.size()

    if h * w <= min_size:
        with torch.no_grad():
            return net(lr.to(device), scale)

    h_half, w_half = h // 2, w // 2
    h_chop, w_chop = h_half + shave, w_half + shave

    patches = [
        lr[..., 0:h_chop, 0:w_chop],
        lr[..., 0:h_chop, w - w_chop:w],
        lr[..., h - h_chop:h, 0:w_chop],
        lr[..., h - h_chop:h, w - w_chop:w],
    ]

    outs = [forward_chop_recursive(net, p, scale, shave, min_size, device) for p in patches]

    h *= scale; w *= scale
    h_half *= scale; w_half *= scale
    h_chop *= scale; w_chop *= scale

    out = torch.zeros((1, outs[0].shape[1], h, w), dtype=outs[0].dtype, device=outs[0].device)
    out[..., 0:h_half, 0:w_half] = outs[0][..., 0:h_half, 0:w_half]
    out[..., 0:h_half, w_half:w] = outs[1][..., 0:h_half, w_chop - w + w_half:w_chop]
    out[..., h_half:h, 0:w_half] = outs[2][..., h_chop - h + h_half:h_chop, 0:w_half]
    out[..., h_half:h, w_half:w] = outs[3][..., h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop]
    return out


# ------------------------------
# 파이프라인
# ------------------------------

def build_model(cfg, device: torch.device):
    module = __import__(f"model.{cfg.model}", fromlist=["Net"])  # model/<name>.py
    net = module.Net(multi_scale=True, group=cfg.group)
    # 래퍼: net(lr, scale) 형태를 보장
    wrapped = ESRGCNNWrapper(net, scale=cfg.scale).to(device)

    state = torch.load(cfg.ckpt_path, map_location="cpu")
    net.load_state_dict(state)

    return wrapped


def inference_lr_mode(net, device, files, cfg):
    """입력이 이미 LR인 경우: x4 업스케일 결과 저장"""
    out_dir = os.path.join(cfg.sample_dir, os.path.splitext(os.path.basename(cfg.ckpt_path))[0], "SR_x4_from_LR")
    os.makedirs(out_dir, exist_ok=True)

    times = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ten = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # 패딩(타일 안정성)
        ten_p, _ = reflect_pad_to_multiple(ten, 1)

        t0 = time.time()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(True, dtype=torch.float16):
                    out = forward_chop_recursive(net, ten_p, scale=cfg.scale, shave=cfg.shave, min_size=cfg.tile_min_size, device=device.type)
            else:
                out = forward_chop_recursive(net, ten_p, scale=cfg.scale, shave=cfg.shave, min_size=cfg.tile_min_size, device=device.type)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.time() - t0)

        # 패딩 되돌리기: LR→SR 이라서 unpad는 필요 없음(입력 pad는 LR 차원)
        # 저장
        out = out.squeeze(0)  # (C, H*scale, W*scale)
        save_tensor_image(out, os.path.join(out_dir, f"{name}_SRx{cfg.scale}.png"))
        print(f"[LR] Saved: {name}")

    if times:
        print(f"LR mode: {len(times)} imgs, avg {sum(times)/len(times):.3f}s")


def inference_hr_mode(net, device, files, cfg):
    """입력이 HR: 1/4 다운 → 모델 x4 → HR과 정렬된 결과 저장(+기준선)"""
    base = os.path.join(cfg.sample_dir, os.path.splitext(os.path.basename(cfg.ckpt_path))[0])
    sr_dir = os.path.join(base, "SR_x4_then_down1_4")  # 최종 HR 해상도 결과
    lr_dir = os.path.join(base, "LR_down1_4")
    ref_nearest_dir = os.path.join(base, "REF_NEAREST_up")
    ref_bicubic_dir = os.path.join(base, "REF_BICUBIC_up")
    for d in [sr_dir, lr_dir, ref_nearest_dir, ref_bicubic_dir]:
        os.makedirs(d, exist_ok=True)

    times = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ten_hr = torch.from_numpy(img).float().div(255.).permute(2,0,1)  # (C,H,W)

        # scale 배수 패딩(다운/업 샘플 일관성 + 타일 안정성)
        pad_b = (cfg.scale - (H % cfg.scale)) % cfg.scale
        pad_r = (cfg.scale - (W % cfg.scale)) % cfg.scale
        ten_hr_p = F.pad(ten_hr.unsqueeze(0), (0, pad_r, 0, pad_b), mode="reflect").squeeze(0)
        Hp, Wp = ten_hr_p.shape[1], ten_hr_p.shape[2]

        # 1) 다운스케일(1/4)
        ten_lr = resize(ten_hr_p, [Hp // cfg.scale, Wp // cfg.scale], interpolation=InterpolationMode.BICUBIC)
        save_tensor_image(ten_lr, os.path.join(lr_dir, f"{name}_LRx1_4.png"))

        # 2) 모델로 x4 업스케일 (타일)
        ten_lr_b = ten_lr.unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(True, dtype=torch.float16):
                    out_sr = forward_chop_recursive(net, ten_lr_b, scale=cfg.scale, shave=cfg.shave, min_size=cfg.tile_min_size, device=device.type)
            else:
                out_sr = forward_chop_recursive(net, ten_lr_b, scale=cfg.scale, shave=cfg.shave, min_size=cfg.tile_min_size, device=device.type)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.time() - t0)

        # 3) HR 해상도로 복귀(패딩 제거)
        sr_x4 = out_sr.squeeze(0)  # (C, 4Hp, 4Wp)
        sr_hr = sr_x4[:, :H, :W]   # 패딩 제거 후 HR과 정렬
        save_tensor_image(sr_hr, os.path.join(sr_dir, f"{name}_SRx4_then_down1_4.png"))

        # 4) 기준선 업샘플(Nearest/Bicubic)
        ref_nearest = resize(ten_lr, [Hp, Wp], interpolation=InterpolationMode.NEAREST)[:, :H, :W]
        ref_bicubic = resize(ten_lr, [Hp, Wp], interpolation=InterpolationMode.BICUBIC)[:, :H, :W]
        save_tensor_image(ref_nearest, os.path.join(ref_nearest_dir, f"{name}_NEAREST_up.png"))
        save_tensor_image(ref_bicubic, os.path.join(ref_bicubic_dir, f"{name}_BICUBIC_up.png"))

        print(f"[HR] Saved: {name}")

    if times:
        print(f"HR mode: {len(times)} imgs, avg {sum(times)/len(times):.3f}s")


# ------------------------------
# Main
# ------------------------------

def main():
    cfg = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(vars(cfg), indent=2))
    print(f"Device: {device}")

    # 모델 빌드/가중치 로드
    net = build_model(cfg, device)

    # 입력 파일 목록
    files = list_images(cfg.test_data_dir)
    if not files:
        raise FileNotFoundError(f"No images found in: {cfg.test_data_dir}")

    # 실행
    if cfg.input_mode == "lr":
        inference_lr_mode(net, device, files, cfg)
    else:
        inference_hr_mode(net, device, files, cfg)


if __name__ == "__main__":
    main()
