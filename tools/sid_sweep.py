r"""批量试听模型内嵌的说话人 id（emb_g 的 109 个音色）。

RVC 的 .pth 里 emb_g 可能有多个说话人嵌入，WebUI 的滑块藏在"模型推理"页
（选完"推理音色"下拉后才会出现）。这个脚本绕过 UI，直接对同一段音频
逐个 sid 推理，方便横向 A/B。

用法（在项目根目录执行）：
    runtime\python.exe tools\sid_sweep.py -i 我的声音.wav
    runtime\python.exe tools\sid_sweep.py -i 我的声音.wav --step 2 --pitch 12
    runtime\python.exe tools\sid_sweep.py -i 我的声音.wav --sids 0,10,20,50,100

输出：sid_sweep_out/sid_000.wav ... 以及 report.txt（含每个文件的时长/响度/亮度）
"""

import argparse
import os
import sys
import time
import traceback

NOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOW_DIR)
os.chdir(NOW_DIR)

# 先把脚本自己的参数取出来，Config 会去 parse sys.argv，别让它看到我们的参数
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-i", "--input", default=None, help="输入音频（wav/mp3/flac）")
parser.add_argument("--pth", default="assets/weights/maoniang/MaoNiang48k.pth")
parser.add_argument("--index", default="assets/weights/maoniang/suoying.index")
parser.add_argument("--pitch", type=int, default=12, help="变调半音数")
parser.add_argument("--method", default="rmvpe", choices=["rmvpe", "fcpe", "pm"])
parser.add_argument("--index-rate", type=float, default=0.35)
parser.add_argument("--protect", type=float, default=0.33)
parser.add_argument("--rms", type=float, default=0.25, help="音量包络混合比例")
parser.add_argument("--sids", default=None, help="逗号分隔，如 0,10,50；优先于 --step")
parser.add_argument("--step", type=int, default=4, help="每隔多少个 sid 取一个")
parser.add_argument("--out", default="sid_sweep_out")
parser.add_argument(
    "--hubert",
    default=None,
    help="hubert 模型目录，如 assets/hubert_chinese；默认官方 hubert_base",
)
args = parser.parse_args()

sys.argv = [sys.argv[0]]
os.environ.setdefault("rmvpe_root", os.path.join(NOW_DIR, "assets", "rmvpe"))

import librosa
import numpy as np
import soundfile as sf
import torch

from configs.config import Config
from infer.hubert import load_hubert_model, hubert_audio_requires_normalization
from infer.rtrvc import get_synthesizer
from infer.vc.pipeline import Pipeline


def find_default_input():
    for root, _, files in os.walk(os.path.join(NOW_DIR, "logs")):
        for name in files:
            if name.lower().endswith(".wav"):
                return os.path.join(root, name)
    return None


def brightness(y, sr):
    """谱质心（Hz），用来快速判断不同 sid 的音色差别。"""
    if y.size < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(y[: 1 << 15] * np.hanning(min(1 << 15, y.size))))
    freqs = np.fft.rfftfreq(min(1 << 15, y.size), 1.0 / sr)
    s = spec.sum()
    return float((spec * freqs).sum() / s) if s > 0 else 0.0


def main():
    wav_path = args.input or find_default_input()
    if not wav_path or not os.path.isfile(wav_path):
        print("[错误] 找不到输入音频，请用 -i 指定")
        return 1

    cfg = Config()
    print("设备：%s  半精度：%s" % (cfg.device, cfg.is_half))
    print("输入：%s" % wav_path)

    net_g, cpt = get_synthesizer(args.pth, cfg.device)
    # get_synthesizer 返回的是 fp32，pipeline 会按 is_half 送 half 特征，这里对齐
    if cfg.is_half:
        net_g = net_g.half()
    tgt_sr = cpt["config"][-1]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    n_spk = cpt["weight"]["emb_g.weight"].shape[0]
    print("模型：%s / %s / %d Hz / if_f0=%s / 说话人数=%d"
          % (os.path.basename(args.pth), version, tgt_sr, if_f0, n_spk))

    if args.sids:
        sids = [int(s) for s in args.sids.split(",") if s.strip() != ""]
    else:
        sids = list(range(0, n_spk, max(1, args.step)))
    sids = [s for s in sids if 0 <= s < n_spk]
    if not sids:
        print("[错误] 没有合法的 sid")
        return 1
    print("待试听 sid：%s" % sids)

    audio = librosa.load(wav_path, sr=16000, mono=True)[0]
    peak = np.abs(audio).max() / 0.95
    if peak > 1:
        audio /= peak

    hubert_dir = args.hubert or None
    if hubert_dir:
        print("[A/B] 使用 hubert 目录：%s" % hubert_dir)
    hubert = load_hubert_model(cfg.device, cfg.is_half, model_dir=hubert_dir)
    hubert_norm = hubert_audio_requires_normalization(hubert_dir)
    print("[A/B] 输入归一化：%s" % hubert_norm)
    pipe = Pipeline(tgt_sr, cfg)

    index_path = args.index if os.path.isfile(args.index or "") else ""
    if not index_path:
        print("[提示] 未找到索引文件，index_rate 不生效")

    out_dir = os.path.join(NOW_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sid in sids:
        t0 = time.perf_counter()
        try:
            out = pipe.pipeline(
                hubert,
                net_g,
                sid,
                audio,
                [0.0, 0.0, 0.0],
                args.pitch,
                args.method,
                index_path,
                args.index_rate,
                if_f0,
                tgt_sr,
                tgt_sr,
                args.rms,
                version,
                args.protect,
                normalize=hubert_norm,
            )
        except Exception:
            print("[失败] sid=%d\n%s" % (sid, traceback.format_exc()))
            continue
        y = out.astype(np.float32) / 32768.0
        path = os.path.join(out_dir, "sid_%03d.wav" % sid)
        sf.write(path, y, tgt_sr)
        cost = time.perf_counter() - t0
        rms = float(np.sqrt(np.mean(y**2)))
        rows.append((sid, cost, len(y) / tgt_sr, rms, brightness(y, tgt_sr)))
        print("sid=%3d  %.2fs  时长 %.2fs  rms %.4f  质心 %.0fHz"
              % (sid, cost, len(y) / tgt_sr, rms, rows[-1][4]))

    if rows:
        report = os.path.join(out_dir, "report.txt")
        with open(report, "w", encoding="utf-8") as f:
            f.write("# sid\ttime\tseconds\trms\tcentroid_hz\n")
            for r in rows:
                f.write("%d\t%.2f\t%.2f\t%.4f\t%.0f\n" % r)
        print("\n输出目录：%s\n汇总：%s" % (out_dir, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
