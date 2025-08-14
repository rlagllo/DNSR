import os
import os.path as osp
import logging
import argparse
import time

import torch
import cv2

from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net


def chop_forward_with_blending(model, x, shave=20, min_size=160000):
    n, c, h, w = x.size()
    if h * w <= min_size:
        return model(x)

    top = slice(0, h // 2 + shave)
    bottom = slice(h - h // 2 - shave, h)
    left = slice(0, w // 2 + shave)
    right = slice(w - w // 2 - shave, w)

    x11 = x[:, :, top, left]
    x12 = x[:, :, top, right]
    x21 = x[:, :, bottom, left]
    x22 = x[:, :, bottom, right]

    y11 = chop_forward_with_blending(model, x11, shave, min_size)
    y12 = chop_forward_with_blending(model, x12, shave, min_size)
    y21 = chop_forward_with_blending(model, x21, shave, min_size)
    y22 = chop_forward_with_blending(model, x22, shave, min_size)

    _, _, h_half, w_half = y11.size()
    h, w = h_half * 2, w_half * 2

    output = x.new_zeros((n, c, h, w))
    count  = x.new_zeros((n, c, h, w))

    output[:, :, :h_half, :w_half] += y11; count[:, :, :h_half, :w_half] += 1
    output[:, :, :h_half, w_half:] += y12; count[:, :, :h_half, w_half:] += 1
    output[:, :, h_half:, :w_half] += y21; count[:, :, h_half:, :w_half] += 1
    output[:, :, h_half:, w_half:] += y22; count[:, :, h_half:, w_half:] += 1

    output /= count
    return output


def parse_args():
    p = argparse.ArgumentParser(description="BSRGAN inference (KAIR) — input as-is, output x4 then downscale to original size")
    p.add_argument("--input", "-i", type=str, required=True, help="입력 이미지(폴더/단일 파일)")
    p.add_argument("--output", "-o", type=str, required=True, help="출력 폴더")
    p.add_argument("--model_path", "-m", type=str, default=osp.join("model_zoo", "best0_G.pth"), help="RRDBNet 가중치(.pth)")
    p.add_argument("--scale", "-s", type=int, default=4, help="모델 업스케일 배수 (보통 4)")
    p.add_argument("--use_tile", action="store_true", help="큰 이미지용 타일/블렌딩 추론")
    p.add_argument("--tile_min_size", type=int, default=640000, help="타일 분할 임계 픽셀 수 (h*w)")
    p.add_argument("--tile_shave", type=int, default=40, help="타일 오버랩")
    p.add_argument("--gpu", type=int, default=0, help="GPU id")
    return p.parse_args()


def main():
    args = parse_args()

    # 로거
    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

    # 디바이스
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"{'GPU ID':>16s} : {torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        logger.info(f"{'GPU ID':>16s} : CPU")

    # 모델
    sf = int(args.scale)
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
    logger.info(f"{'Model Path':>16s} : {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 경로
    in_path, out_path = args.input, args.output
    os.makedirs(out_path, exist_ok=True)
    logger.info(f"{'Input Path':>16s} : {in_path}")
    logger.info(f"{'Output Path':>16s} : {out_path}")
    logger.info(f"{'Scale':>16s} : x{sf}")

    # 입력 목록
    img_paths = util.get_image_paths(in_path) if osp.isdir(in_path) else [in_path]

    for idx, img in enumerate(img_paths, 1):
        img_name, ext = osp.splitext(osp.basename(img))
        logger.info(f"{idx:->4d} --> x{sf} --> {img_name+ext}")

        # 원본 로드 (그대로 모델에 넣음)
        img_ori = util.imread_uint(img, n_channels=3)   # HWC uint8
        h_ori, w_ori = img_ori.shape[:2]
        img_L = util.uint2tensor4(img_ori).to(device)   # BCHW float, [0,1]

        # 추론 (출력 해상도는 보통 4H × 4W)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            if args.use_tile:
                img_E = chop_forward_with_blending(model, img_L, shave=args.tile_shave, min_size=args.tile_min_size)
            else:
                img_E = model(img_L)
        if device.type == "cuda":
            torch.cuda.synchronize()
            max_mem_MB = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            max_mem_MB = 0.0
        logger.info(f"[{img_name}] time: {time.time()-t0:.4f}s, max GPU mem: {max_mem_MB:.2f} MB")

        # 저장: x4 결과를 원본(H×W)로 다운스케일
        img_E = util.tensor2uint(img_E)  # HWC uint8 (대개 4H×4W)
        if img_E.shape[:2] != (h_ori, w_ori):
            # 다운스케일은 보통 INTER_AREA가 깔끔
            img_E = cv2.resize(img_E, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

        util.imsave(img_E, osp.join(out_path, f"{img_name}_RRDB_x{sf}_toOrig.png"))

    logger.info("Done.")


if __name__ == '__main__':
    main()