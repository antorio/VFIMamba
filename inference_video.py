"""
inference_video.py -- VFIMamba video frame interpolation wrapper (v2).

PENTING (v2): resolusi tinggi WAJIB lewat hr_inference() dengan down_scale.
Memakai inference(scale=0) pada video 2K/4K membuat estimasi flow gagal
(flow ~ 0) sehingga frame tengah hanya jadi blending -> terlihat "tidak
terinterpolasi" dan blur. Nilai resmi dari benchmark/XTEST.py:
    2K (2048x1080) -> down_scale = 0.5
    4K (4096x2160) -> down_scale = 0.25

Contoh:
    python inference_video.py --video input.mp4 --multi 2
    python inference_video.py --video input.mp4 --multi 4 --down_scale 0.5
    python inference_video.py --video input.mp4 --multi 4 --down_scale 0.25
    python inference_video.py --video input.mp4 --multi 2 --tta   # ~2x lambat

Catatan:
  --multi N      fps_output = fps_input * N. Tidak harus pangkat 2
                 (timestep arbitrer didukung; lihat benchmark/XTEST_L.py).
  --down_scale   default: auto berdasarkan resolusi. 1.0 = jalur inference()
                 biasa (hanya untuk video kecil, <=1280px).
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=str, required=True)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--model', type=str, default='VFIMamba', choices=['VFIMamba_S', 'VFIMamba'])
    p.add_argument('--multi', type=int, default=2, help='pengali fps (2, 3, 4, ...)')
    p.add_argument('--down_scale', type=float, default=None,
                   help='skala estimasi flow. auto jika kosong. 0.5 utk ~2K, 0.25 utk ~4K')
    p.add_argument('--tta', action='store_true', help='TTA (lebih bagus, ~2x lambat)')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--crf', type=int, default=17)
    p.add_argument('--preset', type=str, default='medium')
    p.add_argument('--audio', action='store_true')
    return p.parse_args()


def auto_down_scale(w, h):
    long_edge = max(w, h)
    if long_edge <= 1280:
        return 1.0
    if long_edge <= 2560:
        return 0.5
    return 0.25


def to_tensor(frame_bgr, device):
    t = torch.from_numpy(frame_bgr.transpose(2, 0, 1)).to(device, non_blocking=True)
    return (t / 255.).unsqueeze(0)


def to_frame(tensor_chw):
    arr = tensor_chw.detach().float().cpu().numpy().transpose(1, 2, 0)
    return (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    assert args.multi >= 2, '--multi minimal 2'
    if not os.path.isfile(args.video):
        sys.exit(f'File tidak ditemukan: {args.video}')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'Gagal membuka video: {args.video}')

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = args.fps if args.fps else src_fps * args.multi

    ds = args.down_scale if args.down_scale is not None else auto_down_scale(width, height)
    use_hr = ds < 1.0

    # hr_inference memperkecil citra dengan faktor ds, dan hasilnya harus tetap
    # habis dibagi 32 -> maka citra ter-pad harus habis dibagi 32/ds.
    divisor = int(round(32 / ds)) if use_hr else 32

    out_path = args.output or f'{os.path.splitext(args.video)[0]}_{args.multi}x.mp4'

    print(f'[info] input : {width}x{height} @ {src_fps:.3f} fps, ~{n_frames} frames')
    print(f'[info] output: {out_path} @ {out_fps:.3f} fps  (multi={args.multi})')
    print(f'[info] model : {args.model} | down_scale={ds} | '
          f'path={"hr_inference" if use_hr else "inference"} | divisor={divisor} | TTA={args.tta}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[warn] CUDA tidak terdeteksi -- akan sangat lambat.')

    model = Model.from_pretrained(args.model)
    model.eval()
    model.device()

    timesteps = [i / args.multi for i in range(1, args.multi)]

    ff = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', f'{out_fps}',
        '-i', '-',
    ]
    if args.audio:
        ff += ['-i', args.video, '-map', '0:v', '-map', '1:a?', '-c:a', 'copy', '-shortest']
    ff += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', str(args.crf), '-preset', args.preset, out_path]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)

    ok, prev = cap.read()
    if not ok:
        sys.exit('Video kosong.')

    padder = InputPadder((1, 3, height, width), divisor=divisor)
    written = 0
    pbar = tqdm(total=max(n_frames - 1, 0), unit='pair', desc='interpolating')

    try:
        with torch.no_grad():
            while True:
                ok, cur = cap.read()
                if not ok:
                    break

                proc.stdin.write(prev.tobytes())
                written += 1

                I0p, I1p = padder.pad(to_tensor(prev, device), to_tensor(cur, device))

                for t in timesteps:
                    if use_hr:
                        mid = model.hr_inference(
                            I0p, I1p, True,
                            TTA=args.tta, fast_TTA=args.tta,
                            timestep=t, down_scale=ds,
                        )
                    else:
                        mid = model.inference(
                            I0p, I1p, True,
                            TTA=args.tta, fast_TTA=args.tta,
                            timestep=t, scale=0,
                        )
                    proc.stdin.write(to_frame(padder.unpad(mid)[0]).tobytes())
                    written += 1

                prev = cur
                pbar.update(1)

        proc.stdin.write(prev.tobytes())
        written += 1
    finally:
        pbar.close()
        cap.release()
        proc.stdin.close()
        proc.wait()

    print(f'[done] {written} frames -> {out_path}')


if __name__ == '__main__':
    main()
