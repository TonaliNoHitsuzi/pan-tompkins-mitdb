"""W2: run Pan-Tompkins on record 100 with three preprocessing configs, report metrics and stage plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
from ecg_lib import BEAT_SYMBOLS, remove_baseline, notch_powerline, pan_tompkins, evaluate

DATA_DIR = r"data/mit-bih-arrhythmia-database-1.0.0"
DOCS = r"docs"
FS = 360
REC = "100"

record = wfdb.rdrecord(f"{DATA_DIR}/{REC}")
ann = wfdb.rdann(f"{DATA_DIR}/{REC}", extension="atr")
ch = record.sig_name.index("MLII")
sig = record.p_signal[:, ch]
ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
print(f"record {REC}: {len(sig)} samples, {len(ref)} reference beats")

configs = {
    "A_raw(PT only)": sig,
    "B_notch60(PT)": notch_powerline(sig, FS, 60.0),
    "C_full(notch+baseline, PT)": remove_baseline(notch_powerline(sig, FS, 60.0), FS),
}

tol = int(0.15 * FS)
results = {}
for name, s in configs.items():
    det = pan_tompkins(s, FS)
    r = evaluate(det, ref, tol)
    results[name] = r
    print(f"{name:34s} TP={r['tp']:5d} FP={r['fp']:4d} FN={r['fn']:4d} "
          f"Se={r['se']*100:.2f}% +P={r['pp']*100:.2f}% F1={r['f1']*100:.2f}%")

sig_c = configs["C_full(notch+baseline, PT)"]
qrs_c, stages = pan_tompkins(sig_c, FS, return_stages=True)

SEC = 10
n = FS * SEC
t = np.arange(n) / FS
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
panels = [
    (sig[:n], "raw MLII (mV)"),
    (stages["filtered"][:n], "1. bandpass 5-15 Hz"),
    (stages["derivative"][:n], "2. derivative"),
    (stages["squared"][:n], "3. squared"),
    (stages["integrated"][:n], "4. moving-window integration (150 ms)"),
]
for ax, (y, label) in zip(axes, panels):
    ax.plot(t, y, linewidth=0.6, color="tab:blue")
    for q in stages["qrs_raw"]:
        if q < n:
            ax.axvline(q / FS, color="red", alpha=0.5, linewidth=0.8)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Pan-Tompkins pipeline stages, record {REC}, first {SEC} s (red = detected QRS)")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{DOCS}/w2_pt_stages_{REC}.png", dpi=150)
print(f"stage plot saved: {DOCS}/w2_pt_stages_{REC}.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=True)
axes[0].plot(t, sig[:n], linewidth=0.6)
axes[0].set_title("before preprocessing (raw MLII)")
axes[1].plot(t, sig_c[:n], linewidth=0.6, color="tab:green")
axes[1].set_title("after notch 60 Hz + baseline removal")
for ax in axes:
    ax.grid(alpha=0.3)
axes[1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(f"{DOCS}/w2_filter_effect_{REC}.png", dpi=150)
print(f"filter plot saved: {DOCS}/w2_filter_effect_{REC}.png")
