import sys, os, time
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.formant_stream import FormantShifter

import sys
y, sr = sf.read(sys.argv[1] if len(sys.argv) > 1 else "input.wav", dtype="float32")
if y.ndim > 1:
    y = y.mean(axis=1)

fs = FormantShifter(alpha=1.16, sr=sr)
BLK = 4800
out = np.zeros_like(y)
t0 = time.perf_counter()
for i in range(0, len(y), BLK):
    seg = y[i : i + BLK]
    out[i : i + BLK] = fs.process(seg)[: len(seg)]
dt = time.perf_counter() - t0
nblk = (len(y) + BLK - 1) // BLK
print("耗时：总 %.1f ms / %d 块 = %.2f ms每块（块时长 %.1f ms，占用 %.1f%%）"
      % (dt * 1000, nblk, dt * 1000 / nblk, BLK / sr * 1000, 100 * (dt / nblk) / (BLK / sr)))

d = np.abs(np.diff(out))
bnd = np.array([i - 1 for i in range(BLK, len(out), BLK)])
print("|diff| 全局 p99=%.4f 最大=%.4f" % (np.percentile(d, 99), d.max()))
print("块边界 |diff| 最大=%.4f 均值=%.5f  vs 全局均值=%.5f"
      % (d[bnd].max(), d[bnd].mean(), d.mean()))

# 与同参数下"不处理"的差异能量，确认不是近似直通
a = out[fs.latency:]
ref = y[: len(a)]
n = min(len(a), len(ref))
print("处理量：输出-输入 差异 rms=%.4f（输入 rms=%.4f）"
      % (np.sqrt(np.mean((a[:n] - ref[:n]) ** 2)), np.sqrt(np.mean(ref[:n] ** 2))))
print("有限性：%s" % np.isfinite(out).all())
