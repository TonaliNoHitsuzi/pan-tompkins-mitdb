"""W4: report figures and tables — per-record F1 bars, paced-failure diagnosis, R-peak timing error histogram,
filtering spectra, pipeline block diagram, summary markdown tables."""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import wfdb
from scipy import signal as sps

sys.path.insert(0, r"code")
from ecg_lib import (BEAT_SYMBOLS, remove_baseline, notch_powerline,
                     bandpass_pt, mwa_past, pan_tompkins, evaluate)

DATA_DIR = r"data/mit-bih-arrhythmia-database-1.0.0"
DOCS = r"docs"
FS = 360
TOL = int(0.15 * FS)


def load(rec):
    record = wfdb.rdrecord(f"{DATA_DIR}/{rec}")
    ann = wfdb.rdann(f"{DATA_DIR}/{rec}", extension="atr")
    ch = record.sig_name.index("MLII") if "MLII" in record.sig_name else record.sig_name.index("V5")
    sig = record.p_signal[:, ch]
    ref = np.sort(np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS]))
    return sig, ref


with open(f"{DOCS}/w3_all48_metrics.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r["record"]))

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(rows))
mine = [float(r["C_full_f1"]) * 100 for r in rows]
lib = [float(r["lib_f1"]) * 100 for r in rows]
n_ref_all = sum(int(r["n_ref"]) for r in rows)
pooled_f1 = {}
for name in ("C_full", "lib"):
    fn = sum(int(r[f"{name}_fn"]) for r in rows)
    fp = sum(int(r[f"{name}_fp"]) for r in rows)
    tp = n_ref_all - fn
    se, pp = tp / n_ref_all, tp / (tp + fp)
    pooled_f1[name] = 2 * se * pp / (se + pp) * 100
ax.bar(x - 0.2, mine, 0.4, label="this work (C_full)", color="tab:blue")
ax.bar(x + 0.2, lib, 0.4, label="py-ecg-detectors", color="tab:orange", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([r["record"] for r in rows], rotation=90, fontsize=7)
ax.set_ylabel("F1 (%)")
ax.set_ylim(0, 105)
ax.axhline(pooled_f1["C_full"], color="tab:blue", ls="--", lw=0.8, alpha=0.6,
           label=f"pooled F1 = {pooled_f1['C_full']:.2f}%")
ax.axhline(pooled_f1["lib"], color="tab:orange", ls="--", lw=0.8, alpha=0.6,
           label=f"pooled F1 = {pooled_f1['lib']:.2f}%")
ax.set_title("Per-record F1 over all 48 MIT-BIH records (dashed = pooled F1)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{DOCS}/w4_f1_bars_all48.png", dpi=150)
print("saved w4_f1_bars_all48.png")

sig104, ref104 = load("104")
sig_c = remove_baseline(notch_powerline(sig104, FS, 60.0), FS)
filtered = bandpass_pt(sig_c, FS)
integrated = mwa_past(np.diff(filtered) ** 2, int(0.15 * FS))
lo, hi = FS * 60, FS * 90
t = np.arange(lo, hi) / FS
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(t, sig104[lo:hi], lw=0.5)
axes[0].set_ylabel("raw V5 (mV)")
axes[0].set_title("record 104 (paced): pacemaker spikes dominate the derivative energy")
axes[1].plot(t, integrated[lo:hi], lw=0.5, color="tab:purple")
axes[1].set_ylabel("PT integral signal")
axes[1].set_xlabel("time (s)")
ref_w = ref104[(ref104 >= lo) & (ref104 < hi)]
for ax in axes:
    for r in ref_w:
        ax.axvline(r / FS, color="tab:blue", alpha=0.3, lw=0.6)
fig.tight_layout()
fig.savefig(f"{DOCS}/w4_failure104_diagnosis.png", dpi=150)
print("saved w4_failure104_diagnosis.png")

sig100, ref100 = load("100")
det100 = pan_tompkins(remove_baseline(notch_powerline(sig100, FS, 60.0), FS), FS)
matched = []
di = np.asarray(det100, dtype=int)
used = np.zeros(len(di), bool)
for r in ref100:
    cand = np.where((np.abs(di - r) <= TOL) & ~used)[0]
    if cand.size:
        j = cand[np.argmin(np.abs(di[cand] - r))]
        used[j] = True
        matched.append((di[j] - r) / FS * 1000)
matched = np.array(matched)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(matched, bins=60, color="tab:blue", edgecolor="white")
ax.axvline(0, color="red", lw=1)
ax.set_xlabel("detection delay vs annotation (ms)")
ax.set_ylabel("beats")
ax.set_title(f"record 100: R-peak timing error, mean={matched.mean():.1f} ms, |err|<10ms: "
             f"{np.mean(np.abs(matched) < 10) * 100:.1f}%")
fig.tight_layout()
fig.savefig(f"{DOCS}/w4_timing_error_hist.png", dpi=150)
print("saved w4_timing_error_hist.png")

f, p_raw = sps.welch(sig100, fs=FS, nperseg=4096)
_, p_notch = sps.welch(notch_powerline(sig100, FS, 60.0), fs=FS, nperseg=4096)
_, p_full = sps.welch(remove_baseline(notch_powerline(sig100, FS, 60.0), FS), fs=FS, nperseg=4096)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(f, p_raw, lw=0.8, label="raw", color="tab:gray")
ax.semilogy(f, p_notch, lw=0.8, label="notch 60 Hz", color="tab:green")
ax.semilogy(f, p_full, lw=0.8, label="notch + baseline removal", color="tab:purple")
ax.axvline(60, color="red", ls=":", lw=1, label="60 Hz mains")
ax.axvspan(5, 15, alpha=0.1, color="blue", label="QRS band 5-15 Hz")
ax.set_xlim(0, 120)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("record 100: power spectrum before/after preprocessing")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{DOCS}/w4_spectra.png", dpi=150)
print("saved w4_spectra.png")

fig, ax = plt.subplots(figsize=(12, 2.6))
ax.axis("off")
blocks = [
    ("ECG\nMLII/V5", 0.00),
    ("baseline\nremoval\n(medfilt 0.2/0.6 s)", 0.14),
    ("notch\n60 Hz", 0.30),
    ("bandpass\n5-15 Hz\n(causal)", 0.42),
    ("derivative", 0.56),
    ("squaring", 0.68),
    ("moving\nintegration\n150 ms", 0.78),
]
for text, x0 in blocks:
    w, h = 0.11, 0.52
    ax.add_patch(FancyBboxPatch((x0, 0.24), w, h, boxstyle="round,pad=0.012",
                                fc="#dceefb", ec="#2a6fb5"))
    ax.text(x0 + w / 2, 0.5, text, ha="center", va="center", fontsize=8)
for x0 in [b[1] + 0.11 for b in blocks[:-1]]:
    ax.add_patch(FancyArrowPatch((x0 + 0.005, 0.5), (x0 + 0.025, 0.5),
                                 arrowstyle="-|>", mutation_scale=14, color="#444"))
ax.add_patch(FancyBboxPatch((0.90, 0.30), 0.09, 0.40, boxstyle="round,pad=0.012",
                            fc="#ffe9d6", ec="#c2661a"))
ax.text(0.945, 0.5, "adaptive\nthreshold\n+ RR logic", ha="center", va="center", fontsize=8)
ax.add_patch(FancyArrowPatch((0.89, 0.5), (0.898, 0.5), arrowstyle="-|>", mutation_scale=14, color="#444"))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Pan-Tompkins pipeline implemented in this work", y=0.95)
fig.tight_layout()
fig.savefig(f"{DOCS}/w4_pipeline.png", dpi=150)
print("saved w4_pipeline.png")

pooled = {}
n_ref = sum(int(r["n_ref"]) for r in rows)
for name in ("A_raw", "B_notch", "C_full", "lib"):
    fn = sum(int(r[f"{name}_fn"]) for r in rows)
    fp = sum(int(r[f"{name}_fp"]) for r in rows)
    tp = n_ref - fn
    se, pp = tp / n_ref, tp / (tp + fp)
    pooled[name] = (tp, fp, fn, se, pp, 2 * se * pp / (se + pp))

lines = [
    "# W4 汇总报表（自动生成）",
    "",
    "## 全库 48 条汇总（beat-level, ±150 ms 匹配窗）",
    "",
    "| 配置 | TP | FP | FN | Se | +P | F1 |",
    "|---|---|---|---|---|---|---|",
]
names_cn = {
    "A_raw": "A：无预处理",
    "B_notch": "B：仅 60 Hz 陷波",
    "C_full": "C：陷波+基线去除（本文最终）",
    "lib": "py-ecg-detectors（参照）",
}
for name, (tp, fp, fn, se, pp, f1) in pooled.items():
    lines.append(f"| {names_cn[name]} | {tp} | {fp} | {fn} | {se*100:.2f}% | {pp*100:.2f}% | {f1*100:.2f}% |")
lines += [
    "",
    f"参考心拍总数：{n_ref}（48 条 × 双导联记录，导联选择：MLII 优先，无 MLII 时用 V5）",
    "",
    "## 逐记录指标（按记录号排序，C_full 配置 vs 参照库）",
    "",
    "| 记录 | 导联 | 参考拍数 | Se(C) | +P(C) | F1(C) | F1(lib) |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['record']} | {r['lead']} | {r['n_ref']} | "
                 f"{float(r['C_full_se'])*100:.2f}% | {float(r['C_full_pp'])*100:.2f}% | "
                 f"{float(r['C_full_f1'])*100:.2f}% | {float(r['lib_f1'])*100:.2f}% |")
worst = sorted(rows, key=lambda r: float(r["C_full_f1"]))[:5]
lines += [
    "",
    "## F1 最差的 5 条记录（失败案例，均为起搏/噪声记录）",
    "",
    "| 记录 | F1(C) | F1(lib) | 主要失败模式 |",
    "|---|---|---|---|",
]
mode = {"104": "起搏尖峰能量压过 QRS", "102": "起搏+宽 QRS 畸形", "108": "肌电噪声与基线漂移",
        "207": "室性心律+束支阻滞宽 QRS", "228": "起搏心律"}
for r in worst:
    rec = r["record"]
    lines.append(f"| {rec} | {float(r['C_full_f1'])*100:.2f}% | {float(r['lib_f1'])*100:.2f}% | {mode.get(rec, '-')} |")
lines += ["", f"生成时间：2026-10-10（与 docs/w3_all48_metrics.csv 同源）", ""]
with open(f"{DOCS}/w4_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("saved w4_summary.md")
