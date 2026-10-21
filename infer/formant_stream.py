"""流式共振峰搬移（formant shift / vocal tract length）。

与 RVC 自带"性别因子"的区别：那个是 f0 反移 + 输出重采样，两者互相抵消，实测包络
缩放 ≈1.00，只把高频提亮。这里只沿频率轴缩放谱包络（E'(f)=E(f/alpha)），f0 与相位
都不动，故不会引入颤音、也不会改变音高。

数学：logE(f) = Σ c[n]·cos(2πfn/Fs)，要 logE'(f) = logE(f/alpha) 只需
      c'[m] = alpha · c[alpha·m]
（alpha 因子来自离散求和的步长换算；漏掉它会让 log 包络整体被压缩 1/alpha，
 增益退化成全频带常数偏移，听感上几乎没变化——这是最初踩的坑。）

流式实现：STFT(hop 50% 重叠, sqrt-Hann 分析/合成窗) + 逐帧变换 + OLA。
附加延迟 = n_fft - hop = 21.3ms @48k/2048/1024。

工程上必须做的三件事（否则"α 越大越毛"）：
1. 包络沿时间平滑（逐帧独立估包络时，增益逐频点抖动实测 2.0dB/p95 6.2dB）；
2. 软膝盖限幅 max·tanh(x/max)，硬裁剪会让 7~20% 的频点变成移动的平顶陷波；
3. 高频收敛（默认 3.5kHz 起衰减到 0.35 倍），避免 5~6kHz 齿音凸起。

离线原型见 tools/formant_warp.py（同一个数学，整段处理，用于 A/B 试听）。
"""

import numpy as np

__all__ = ["ceps_of", "env_from_ceps", "FormantShifter"]


def ceps_of(logmag, n_fft):
    """log 幅度谱 (bins, T) -> 实倒谱前半 (bins, T)。"""
    full = np.fft.irfft(logmag, n=n_fft, axis=0)
    return full[: n_fft // 2 + 1]


def env_from_ceps(c, n_fft):
    """截断倒谱 (L, T) -> log 谱包络 (bins, T)（偶对称补齐后 rfft）。"""
    T = c.shape[1]
    L = c.shape[0]
    C = np.zeros((n_fft, T), dtype=c.dtype)
    C[:L] = c
    if L > 1:
        C[n_fft - L + 1 :] = c[1:L][::-1]
    return np.fft.rfft(C, n=n_fft, axis=0).real


def _sqrt_hann(n):
    return np.sqrt(np.hanning(n + 1)[:n])


class FormantShifter:
    """逐块调用的小工具：输入/输出等长，内部固定延迟 lat 个采样。"""

    def __init__(self, alpha=1.16, sr=48000, n_fft=1024, hop=None, lifter=88,
                 max_gain_db=12.0, hf_taper_hz=3500.0, hf_depth=0.65,
                 env_smooth=5, gate_db=35.0, gate_range=15.0, peak_release_db_s=6.0):
        self.alpha = float(np.clip(alpha, 0.85, 1.35))
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop or n_fft // 2)
        self.lifter = int(lifter)
        self.max_gain_db = float(max_gain_db)
        self.hf_taper_hz = float(hf_taper_hz)
        self.hf_depth = float(hf_depth)
        self.env_smooth = max(1, int(env_smooth))
        self.gate_db = float(gate_db)
        self.gate_range = float(gate_range)
        self.peak_release = float(peak_release_db_s) * self.hop / float(self.sr)

        self.win = _sqrt_hann(self.n_fft).astype(np.float64)
        # 频率轴与高频收敛系数（缓存，避免每帧重算）
        self.freqs = np.linspace(0.0, self.sr / 2.0, self.n_fft // 2 + 1)
        if self.hf_taper_hz > 0:
            t = np.clip((self.freqs - self.hf_taper_hz)
                        / max(self.sr / 2.0 - self.hf_taper_hz, 1.0), 0.0, 1.0)
            self.hf_factor = 1.0 - self.hf_depth * t ** 1.5
        else:
            self.hf_factor = None
        self.reset()
        # 历史倒谱按"最大 alpha"固定长度存，否则热切换 alpha 时 L_src 变化会让
        # 缓冲里出现不同长度的数组，np.mean 直接报错（拖滑块必崩）。
        _L = min(self.lifter, self.n_fft // 2 - 1)
        self._L_max = min(int(np.ceil(max(_L, 4) * 1.35)) + 4, self.n_fft // 2 - 1)

    # ------------------------------------------------------------------ #
    @property
    def latency(self):
        """额外延迟（采样点）= n_fft。

        构成：OLA 本身的 n_fft-hop（第 m 帧的头部对应 n_fft-hop 个采样之前的输入）
        + 输出 FIFO 预填的 hop（吸收不足一帧的余量，保证进出等长、不欠流）。
        实测 n_fft=1024/hop=512 时逐样本恒等误差 0.000000，延迟 1024 采样 = 21.3ms。
        """
        return self.n_fft

    def reset(self):
        self.frame = np.zeros(self.n_fft, dtype=np.float64)
        self.acc = np.zeros(self.n_fft, dtype=np.float64)
        self.outq = np.zeros(self.hop, dtype=np.float64)
        self.inbuf = np.zeros(0, dtype=np.float64)
        self._ceps_hist = []
        self._peak_db = -120.0

    # ------------------------------------------------------------------ #
    def _constants(self):
        """按当前 alpha 预计算倒谱重采样索引。"""
        L = min(self.lifter, self.n_fft // 2 - 1)
        L = max(L, 4)
        L_src = min(int(np.ceil(L * max(self.alpha, 1.0))) + 4, self.n_fft // 2 - 1)
        L_src = max(L_src, L + 2)
        idx = np.arange(L, dtype=np.float64) * self.alpha
        base = np.arange(L_src, dtype=np.float64)
        return L, L_src, idx, base

    def _frame_gain(self, mag):
        """单帧：返回 (bins,) 的线性增益。"""
        logmag = np.log(mag + 1e-9)[:, None]
        c_full = ceps_of(logmag, self.n_fft)[:, 0]

        if self.env_smooth > 1:
            self._ceps_hist.append(c_full[: self._L_max].copy())
            if len(self._ceps_hist) > self.env_smooth:
                self._ceps_hist.pop(0)
            c_s = np.mean(self._ceps_hist, axis=0)
        else:
            self._ceps_hist = []
            c_s = c_full

        if abs(self.alpha - 1.0) < 1e-6:
            return None  # 直通

        L, L_src, idx, base = self._constants()
        c_src = c_s[:L_src]
        c_new = (self.alpha * np.interp(idx, base, c_src))[:, None]
        delta = (env_from_ceps(c_new, self.n_fft) - env_from_ceps(c_s[:L, None], self.n_fft))[:, 0]
        delta -= delta.mean()
        raw_db = delta * (20.0 / np.log(10.0))
        gain_db = self.max_gain_db * np.tanh(raw_db / self.max_gain_db)

        if self.hf_factor is not None:
            gain_db = gain_db * self.hf_factor

        # 能量门控：弱帧（气声、静音）包络估计不可靠，增益退回 1.0
        frame_db = 10.0 * np.log10(float(np.mean(mag ** 2)) + 1e-20)
        self._peak_db = max(frame_db, self._peak_db - self.peak_release)
        w = (frame_db - (self._peak_db - self.gate_db)) / self.gate_range
        w = float(np.clip(w, 0.0, 1.0))
        gain_db = gain_db * w

        return np.power(10.0, gain_db / 20.0)

    def _process_frame(self):
        spec = np.fft.rfft(self.frame * self.win)
        mag = np.abs(spec)
        gain = self._frame_gain(mag)
        if gain is None:
            y = self.frame * self.win
        else:
            y = np.fft.irfft(spec * gain, n=self.n_fft)
        return y * self.win  # 合成窗；分析×合成 = Hann，hop=50% 时 OLA 和恒为 1

    # ------------------------------------------------------------------ #
    def process(self, x):
        """输入一维 float 数组，返回等长数组（内部延迟 latency 个采样）。"""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        self.inbuf = np.concatenate([self.inbuf, x]) if self.inbuf.size else x.copy()
        while self.inbuf.shape[0] >= self.hop:
            self.frame = np.concatenate([self.frame[self.hop:], self.inbuf[: self.hop]])
            self.inbuf = self.inbuf[self.hop:]
            self.acc += self._process_frame()
            self.outq = np.concatenate([self.outq, self.acc[: self.hop]])
            self.acc = np.concatenate([self.acc[self.hop:], np.zeros(self.hop)])

        if self.outq.shape[0] >= n:
            out = self.outq[:n].copy()
            self.outq = self.outq[n:]
        else:
            out = np.zeros(n, dtype=np.float64)
            out[: self.outq.shape[0]] = self.outq
            self.outq = self.outq[:0]
        return out.astype(np.float32)
