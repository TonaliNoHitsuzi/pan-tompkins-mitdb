"""W1 验证脚本：用 wfdb 读取 MIT-BIH 100 号记录，画信号 + R 波标注。"""
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"data/mit-bih-arrhythmia-database-1.0.0"
OUT_FIG = r"docs/w1_record100.png"

FS = 360
SEC = 10  # 画前 10 秒

record = wfdb.rdrecord(f"{DATA_DIR}/100")
ann = wfdb.rdann(f"{DATA_DIR}/100", extension="atr")

print(f"记录: {record.record_name}")
print(f"采样率: {record.fs} Hz, 导联: {record.sig_name}, 时长: {record.sig_len / record.fs / 60:.1f} min")
beat_sym = ["N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"]
r_mask = [s in beat_sym for s in ann.symbol]
r_samples = ann.sample[r_mask]
print(f"总心拍标注数: {sum(r_mask)} / 全部标注 {len(ann.sample)}")

n = FS * SEC
t = [i / FS for i in range(n)]
r_in_win = r_samples[(r_samples >= 0) & (r_samples < n)]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ch in range(2):
    axes[ch].plot(t, record.p_signal[:n, ch], linewidth=0.6, color="tab:blue")
    axes[ch].scatter(r_in_win / FS, record.p_signal[r_in_win, ch], color="red", zorder=3, s=18)
    axes[ch].set_ylabel(f"{record.sig_name[ch]} (mV)")
    axes[ch].grid(alpha=0.3)
axes[1].set_xlabel("Time (s)")
axes[0].set_title("MIT-BIH Record 100, first 10 s (red = annotated R peaks)")
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
print(f"图已保存: {OUT_FIG}")
print(f"窗口内心拍数: {len(r_in_win)}, 平均心率: {len(r_in_win) / SEC * 60:.1f} bpm")
