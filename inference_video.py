"""
inference_video.py -- VFIMamba video frame interpolation wrapper.

Contoh:
    python inference_video.py --video input.mp4 --multi 2
    python inference_video.py --video input.mp4 --multi 3 --model VFIMamba
    python inference_video.py --video input.mp4 --multi 2 --scale 0.5   # untuk 2K
    python inference_video.py --video input.mp4 --multi 4 --tta         # kualitas max, ~2x lambat

Catatan:
  --multi N  = jumlah frame output per frame input (N-1 frame baru per pasangan).
               fps_output = fps_input * N. Tidak harus pangkat 2.
  --scale    = 0 (auto/full). Pakai 0.5 untuk 2K, 0.25 untuk 4K bila VRAM/kualitas flow bermasalah.
  --tta      = mati secara default (TTA menggandakan waktu proses).
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
    p.add_argument('--video', type=str, required=True, help='path video input')
    p.add_argument('--output', type=str, default=None, help='path video output (default: <nama>_Nx.mp4)')
    p.add_argument('--model', type=str, default='VFIMamba', choices=['VFIMamba_S', 'VFIMamba'])
    p.add_argument('--multi', type=int, default=2, help='pengali fps (2 = 2x, 3 = 3x, dst)')
    p.add_argument('--scale', type=float, default=0.0, help='0=full, 0.5 utk 2K, 0.25 utk 4K')
    p.add_argument('--tta', action='store_true', help='aktifkan TTA (lebih bagus, ~2x lebih lambat)')
    p.add_argument('--fps', type=float, default=None, help='override fps output')
    p.add_argument('--crf', type=int, default=17, help='kualitas x264 (makin kecil makin bagus)')
    p.add_argument('--preset', type=str, default='medium')
    p.add_argument('--audio', action='store_true', help='ikutkan audio dari video asli')
    p.add_argument('--fp16', action='store_true', help='inferensi half precision (eksperimental)')
    return p.parse_args()


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

    out_path = args.output or f'{os.path.splitext(args.video)[0]}_{args.multi}x.mp4'

    print(f'[info] input : {width}x{height} @ {src_fps:.3f} fps, ~{n_frames} frames')
    print(f'[info] output: {out_path} @ {out_fps:.3f} fps  (multi={args.multi})')
    print(f'[info] model : {args.model} | scale={args.scale} | TTA={args.tta} | fp16={args.fp16}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[warn] CUDA tidak terdeteksi -- akan sangat lambat.')

    model = Model.from_pretrained(args.model)
    model.eval()
    model.device()

    # timestep intermediate: i/multi untuk i = 1..multi-1
    timesteps = [i / args.multi for i in range(1, args.multi)]

    ff = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', f'{out_fps}',
        '-i', '-',
    ]
    if args.audio:
        ff += ['-i', args.video, '-map', '0:v', '-map', '1:a?', '-c:a', 'copy', '-shortest']
    ff += [
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', str(args.crf), '-preset', args.preset,
        out_path,
    ]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)

    ok, prev = cap.read()
    if not ok:
        sys.exit('Video kosong.')

    padder = InputPadder((1, 3, height, width), divisor=32)
    written = 0
    pbar = tqdm(total=max(n_frames - 1, 0), unit='pair', desc='interpolating')

    autocast = torch.autocast('cuda', dtype=torch.float16) if (args.fp16 and device.type == 'cuda') \
        else torch.autocast('cuda', enabled=False)

    try:
        with torch.no_grad():
            while True:
                ok, cur = cap.read()
                if not ok:
                    break

                proc.stdin.write(prev.tobytes())
                written += 1

                I0 = to_tensor(prev, device)
                I1 = to_tensor(cur, device)
                I0p, I1p = padder.pad(I0, I1)

                for t in timesteps:
                    with autocast:
                        mid = model.inference(
                            I0p, I1p, True,
                            TTA=args.tta, fast_TTA=args.tta,
                            timestep=t, scale=args.scale,
                        )
                    proc.stdin.write(to_frame(padder.unpad(mid)[0]).tobytes())
                    written += 1

                prev = cur
                pbar.update(1)

        # frame terakhir
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
