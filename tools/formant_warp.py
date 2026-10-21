"""离线共振峰搬移（formant shift / vocal tract length）原型。

原理
----
把谱包络沿频率轴缩放 alpha 倍：  E'(f) = E(f / alpha)
等价于把声道长度乘以 1/alpha，共振峰整体上移。

实现用**倒谱重采样**而不是"LPC 曲线插值 + 增益乘回"：
  log E(f) = sum_n c[n] * cos(2*pi*f*n/Fs)        (实倒谱，偶对称)
  要求   log E'(f) = log E(f/alpha)
       = sum_n c[n] * cos(2*pi*f*n/(alpha*Fs))
       = sum_m c'[m] * cos(2*pi*f*m/Fs)
  只要  c'[m] = c[m/alpha]                        (重采样倒谱轴)
这是精确等式，插值误差只来自 c 的平滑性，没有增益裁剪、没有比例过冲。

与 RVC 自带"性别因子"的区别：那个是 f0 反移 + 输出重采样，两者互相抵消，
实测包络缩放 ≈1.00，只是把高频提亮。这里只搬包络、f0 与相位都不动。

用法：
    runtime\\python.exe tools\\formant_warp.py -i in.wav
    runtime\\python.exe tools\\formant_warp.py -i in.wav --alphas 1.0,1.08,1.16,1.24 -o out
    runtime\\python.exe tools\\formant_warp.py --selftest          # 合成元音验证比例
"""

import argparse
import os
import sys

import numpy as np

NOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOW_DIR)

import librosa
import soundfile as sf



# --------------------------------------------------------------------------- #
# 倒谱：log 谱 <-> 实倒谱
# --------------------------------------------------------------------------- #
def _ceps(logmag, n_fft):
    """(n_fft//2+1, T) 的 log 幅度谱 -> (n_fft//2+1, T) 实倒谱（取前一半）。"""
    full = np.fft.irfft(logmag, n=n_fft, axis=0)  # (n_fft, T)
    return full[: n_fft // 2 + 1]


def _from_ceps(c, n_fft):
    """截断倒谱 -> log 谱包络 (n_fft//2+1, T)。"""
    T = c.shape[1]
    L = c.shape[0]
    C = np.zeros((n_fft, T), dtype=c.dtype)
    C[:L] = c
    if L > 1:
        C[n_fft - L + 1 :] = c[1:L][::-1]  # 偶对称补齐
    return np.fft.rfft(C, n=n_fft, axis=0).real


def envelope(y, sr, n_fft=2048, hop=512, lifter=64):
    """逐帧倒谱包络（幅值，非 dB），形状 (n_fft//2+1, T)。"""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    logmag = np.log(np.abs(S) + 1e-9)
    c = _ceps(logmag, n_fft)[: max(4, min(lifter, n_fft // 2 - 1))].copy()
    return np.exp(_from_ceps(c, n_fft))


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #
def warp(y, sr, alpha, n_fft=2048, hop=512, lifter=64,
         max_gain_db=12.0, target_rms=0.12, env_smooth=5, smooth_ms=0.0,
         hf_taper_hz=3500.0, hf_depth=0.65):
    """只沿频率轴缩放谱包络（共振峰整体上移 alpha 倍），f0/相位不变。

    env_smooth : 包络的时间平滑帧数（奇数）。逐帧独立估包络会让增益逐频点抖动，
                 实测帧间跳变 2.0dB/p95 6.2dB → 听感"毛、不连续"，平滑后降到 0.8/2.4。
    smooth_ms  : 增益的时间一阶平滑（毫秒），0=关。
    """
    if abs(alpha - 1.0) < 1e-6:
        out = y.astype(np.float64).copy()
        if target_rms:
            out *= float(target_rms) / np.sqrt(np.mean(out ** 2) + 1e-12)
        return out.astype(np.float32)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(S) + 1e-9
    phase = np.angle(S)
    logmag = np.log(mag)

    L = max(4, min(lifter, n_fft // 2 - 1))
    # c'[m] = c[alpha*m]：需要读到更远的倒谱系数，故源截断长度放大 alpha 倍
    L_src = min(int(np.ceil(L * max(alpha, 1.0))) + 4, n_fft // 2 - 1)
    L_src = max(L_src, L + 2)
    c = _ceps(logmag, n_fft)[:L_src].copy()       # (L_src, T)
    if env_smooth and env_smooth > 1:
        # 包络沿时间轴平滑：逐帧独立估计会让增益逐频点抖动，是"毛"的主要来源
        ker = np.ones(int(env_smooth)) / float(env_smooth)
        c = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, c)

    idx = np.arange(L, dtype=np.float64) * float(alpha)
    base = np.arange(L_src, dtype=np.float64)
    c_new = np.empty((L, c.shape[1]), dtype=c.dtype)
    for t in range(c.shape[1]):
        # c'[m] = alpha * c[alpha*m]：alpha 因子来自离散求和的步长换算，
        # 漏掉它会让 log 包络整体被压缩 1/alpha，增益变成全频带常数偏移，倾斜只剩一点点。
        c_new[:, t] = alpha * np.interp(idx, base, c[:, t])

    log_env_old = _from_ceps(c[:L], n_fft)
    log_env_new = _from_ceps(c_new, n_fft)

    delta = log_env_new - log_env_old
    delta -= delta.mean(axis=0, keepdims=True)  # 去掉每帧整体增益，避免电平跳动
    raw_db = delta * (20.0 / np.log(10.0))
    # 软膝盖而非硬裁剪：硬裁剪会让大量频点变成移动的平顶陷波（α=1.24 时 20% 频点），
    # 这是颗粒感的第二大来源。tanh 保留大小顺序、峰仍会位移，但没有平顶。
    gain_db = max_gain_db * np.tanh(raw_db / max_gain_db)

    # 能量门控：弱帧（气声、尾部、静音）的包络估计不可靠，实测 6.48s 一处弱帧
    # 被 +14dB 放大。这些帧把增益平滑退回 1.0，不做处理。
    frame_db = 10.0 * np.log10((mag ** 2).mean(axis=0) + 1e-20)
    ref = np.percentile(frame_db, 90)
    w = np.clip((frame_db - (ref - 35.0)) / 15.0, 0.0, 1.0)
    if len(w) >= 5:  # 3 帧滑动平均，避免增益忽开忽关
        k = np.ones(5) / 5.0
        w = np.convolve(w, k, mode="same")
    gain_db = gain_db * w[None, :]

    if smooth_ms and smooth_ms > 0:  # 增益时间一阶平滑（因果，实时可直接照搬）
        a = np.exp(-hop / float(sr) / (float(smooth_ms) / 1000.0))
        sm = np.empty_like(gain_db)
        sm[:, 0] = gain_db[:, 0]
        for t in range(1, gain_db.shape[1]):
            sm[:, t] = a * sm[:, t - 1] + (1.0 - a) * gain_db[:, t]
        gain_db = sm

    if hf_taper_hz and hf_taper_hz > 0:
        # 高频收敛：说话人身份主要落在 F1~F3（300~3500Hz），4kHz 以上的"搬移"听感上
        # 主要是齿音变亮。素材 4.7kHz 的包络峰被搬到 5.5kHz 后带来 +6~8dB 凸起，
        # 实测是"α 越大越毛"的主因，故对高频段的增益做渐进收敛。
        freqs = np.linspace(0.0, sr / 2.0, gain_db.shape[0])
        t = np.clip((freqs - float(hf_taper_hz)) / max(sr / 2.0 - float(hf_taper_hz), 1.0), 0.0, 1.0)
        gain_db = gain_db * (1.0 - float(hf_depth) * t ** 1.5)[:, None]

    gain = 10.0 ** (gain_db / 20.0)

    S_new = mag * gain * np.exp(1j * phase)
    out = librosa.istft(S_new, hop_length=hop, n_fft=n_fft, window="hann",
                        length=len(y))

    if target_rms:
        # 统一到同一 RMS：A/B 时音量一致，否则"更响"会被误听成"更好"
        cur = np.sqrt(np.mean(out ** 2) + 1e-12)
        out *= float(target_rms) / cur
    peak = float(np.abs(out).max())
    if peak > 0.985:
        out *= 0.985 / peak
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# 校验：共振峰估计
# --------------------------------------------------------------------------- #
def ltas_peaks(y, sr, bands=((300, 1100), (1100, 2200), (2200, 4200)), lifter=96):
    """时间平均倒谱包络在各频带内的峰位置（Hz），用于核对包络搬移量。

    注意不能直接对原始谱取峰：浊音的 LTAS 是 f0 的梳状谱，带内 argmax 会被
    最近的谐波钉住（实测 f0=200 时 300-1100 带永远读 800）。所以先做倒谱
    截断得到平滑包络，再在包络上取带内极值。
    """
    S = librosa.stft(y, n_fft=2048, hop_length=512, window="hann", center=True)
    logmag = np.log(np.abs(S) + 1e-9)
    L = max(4, min(lifter, 2047))
    env = _from_ceps(_ceps(logmag, 2048)[:L], 2048)
    T = env.shape[1]
    avg = env[:, T // 4 : 3 * T // 4].mean(axis=1)
    freqs = np.linspace(0.0, sr / 2.0, env.shape[0])
    out = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs <= hi)
        out.append(round(float(freqs[m][int(np.argmax(avg[m]))])) if m.any() else 0)
    return out


def _synth_vowel(sr, dur=1.2, f0=200.0, fmts=(700, 1220, 2600), bws=(90, 110, 160)):
    """合成一个稳定的测试元音（前三个共振峰的级联谐振器）。"""
    n = int(sr * dur)
    src = np.zeros(n)
    period = int(round(sr / f0))
    src[::period] = 1.0  # 脉冲串，谐波足够密，否则 LPC 只能看到谐波
    y = src.copy()
    for f, bw in zip(fmts, bws):
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(th), r * r]
        y = _lfilter([1.0 - r], a, y)
    y *= 0.9 / (np.abs(y).max() + 1e-9)
    return y.astype(np.float32)


def _lfilter(b, a, x):
    from scipy.signal import lfilter
    return lfilter(b, a, x)


def selftest():
    sr = 48000
    y = _synth_vowel(sr)
    base = ltas_peaks(y, sr)
    print("合成元音  原始共振峰 %s" % base)
    print("alpha | 实测带内峰 | 首峰比值 | 目标")
    for a in (1.05, 1.08, 1.12, 1.16, 1.20, 1.24, 1.30):
        out = warp(y, sr, a, lifter=88, env_smooth=args.env_smooth)
        fs = ltas_peaks(out, sr)
        ratio = fs[0] / base[0] if fs and base else float("nan")
        print("%.2f  | %s | %.3f | %.3f" % (a, fs, ratio, a))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default=None)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--alphas", default="1.0,1.08,1.16,1.24")
    p.add_argument("--lifter", type=int, default=64,
                   help="倒谱截断长度，越大包络越细（48k 下 64≈750Hz 分辨率）")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--max-gain-db", type=float, default=12.0)
    p.add_argument("--target-rms", type=float, default=0.12)
    p.add_argument("--env-smooth", type=int, default=5, help="包络时间平滑帧数(奇数)，1=关")
    p.add_argument("--smooth-ms", type=float, default=0.0, help="增益一阶平滑毫秒，0=关")
    p.add_argument("--hf-taper", type=float, default=3500.0, help="高频收敛起点Hz，0=关")
    p.add_argument("--hf-depth", type=float, default=0.65, help="高频收敛深度(0~1)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        p.error("需要 -i 输入文件，或用 --selftest")

    y, sr = sf.read(args.input, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    out_dir = args.out or os.path.join(os.path.dirname(args.input), "warp")
    os.makedirs(out_dir, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    print("输入 %s  %.2fs  %d Hz" % (os.path.basename(args.input), len(y) / sr, sr))
    base = ltas_peaks(y, sr)
    print("alpha | 文件 | 带内峰(Hz) | 首峰比值")
    for a in alphas:
        out = warp(y, sr, a, n_fft=args.n_fft, hop=args.hop,
                   lifter=args.lifter, max_gain_db=args.max_gain_db,
                   target_rms=args.target_rms, env_smooth=args.env_smooth,
                   smooth_ms=args.smooth_ms, hf_taper_hz=args.hf_taper,
                   hf_depth=args.hf_depth)
        name = "warp_a%03d.wav" % round(a * 100)
        sf.write(os.path.join(out_dir, name), out, sr)
        fs = ltas_peaks(out, sr)
        ratio = (fs[0] / base[0]) if base and fs and base[0] else float("nan")
        print("%.2f  | %s | %s -> %s | %.2f" % (a, name, base, fs, ratio))
    print("输出目录：%s" % out_dir)


if __name__ == "__main__":
    main()
