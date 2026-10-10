"""W3: batch evaluation over all 48 MIT-BIH records, three preprocessing configs + py-ecg-detectors baseline,
per-record CSV, pooled metrics, worst-case failure plots."""
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
from ecgdetectors import Detectors
from ecg_lib import BEAT_SYMBOLS, remove_baseline, notch_powerline, pan_tompkins, evaluate

DATA_DIR = r"data/mit-bih-arrhythmia-database-1.0.0"
DOCS = r"docs"
FS = 360
TOL = int(0.15 * FS)

with open(f"{DATA_DIR}/RECORDS", encoding="utf-8") as f:
    records = [ln.strip() for ln in f if ln.strip()]
print(f"{len(records)} records loaded from RECORDS")


def pick_channel(sig_names):
    for name in ("MLII", "V5", "V1", "V2"):
        if name in sig_names:
            return sig_names.index(name)
    return 0


def load_record(rec):
    record = wfdb.rdrecord(f"{DATA_DIR}/{rec}")
    ann = wfdb.rdann(f"{DATA_DIR}/{rec}", extension="atr")
    ch = pick_channel(record.sig_name)
    sig = record.p_signal[:, ch]
    ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    lead = record.sig_name[ch]
    return sig, ref, lead


detectors = Detectors(FS)
rows = []
t0 = time.time()
for rec in records:
    sig, ref, lead = load_record(rec)
    cfgs = {
        "A_raw": sig,
        "B_notch": notch_powerline(sig, FS, 60.0),
        "C_full": remove_baseline(notch_powerline(sig, FS, 60.0), FS),
    }
    row = dict(record=rec, lead=lead, n_ref=len(ref))
    for name, s in cfgs.items():
        r = evaluate(pan_tompkins(s, FS), ref, TOL)
        row[f"{name}_se"] = r["se"]
        row[f"{name}_pp"] = r["pp"]
        row[f"{name}_f1"] = r["f1"]
        row[f"{name}_fn"] = r["fn"]
        row[f"{name}_fp"] = r["fp"]
    det_lib = np.asarray(detectors.pan_tompkins_detector(sig))
    r = evaluate(det_lib, ref, TOL)
    row["lib_se"] = r["se"]
    row["lib_pp"] = r["pp"]
    row["lib_f1"] = r["f1"]
    row["lib_fn"] = r["fn"]
    row["lib_fp"] = r["fp"]
    rows.append(row)
    print(f"{rec} [{lead}] ref={len(ref):6d} | "
          f"C: Se={row['C_full_se']*100:6.2f}% +P={row['C_full_pp']*100:6.2f}% F1={row['C_full_f1']*100:6.2f}% | "
          f"lib F1={row['lib_f1']*100:6.2f}%")

print(f"elapsed {time.time()-t0:.0f}s")

fields = list(rows[0].keys())
with open(r"docs/w3_all48_metrics.csv", "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("csv saved: docs/w3_all48_metrics.csv")

print("\n=== pooled (summed TP/FP/FN over 48 records) ===")
n_ref = sum(row["n_ref"] for row in rows)
for name in ("A_raw", "B_notch", "C_full", "lib"):
    fn = sum(row[f"{name}_fn"] for row in rows)
    fp = sum(row[f"{name}_fp"] for row in rows)
    tp = n_ref - fn
    se, pp = tp / n_ref, tp / (tp + fp)
    print(f"{name:8s} beats={n_ref} TP={tp} FP={fp} FN={fn} "
          f"Se={se*100:.2f}% +P={pp*100:.2f}% F1={2*se*pp/(se+pp)*100:.2f}%")
worst = sorted(rows, key=lambda r: r["C_full_f1"])[:3]
print("\nworst 3 by C_full F1:", [(r["record"], round(r["C_full_f1"] * 100, 2)) for r in worst])

for row in worst:
    rec = row["record"]
    sig, ref, lead = load_record(rec)
    sig_c = remove_baseline(notch_powerline(sig, FS, 60.0), FS)
    det = pan_tompkins(sig_c, FS)
    lo, hi = 0, FS * 30
    t = np.arange(lo, hi) / FS
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, sig[lo:hi], linewidth=0.5, color="tab:blue", label="raw")
    ax.plot(t, sig_c[lo:hi], linewidth=0.5, color="tab:green", alpha=0.6, label="filtered")
    ax.scatter(ref[(ref >= lo) & (ref < hi)] / FS, np.interp(ref[(ref >= lo) & (ref < hi)], np.arange(len(sig)), sig),
               color="blue", marker="x", s=60, zorder=4, label="reference R")
    dw = det[(det >= lo) & (det < hi)]
    ax.scatter(dw / FS, np.interp(dw, np.arange(len(sig)), sig), facecolors="none",
               edgecolors="red", s=80, zorder=5, label="detected R")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"record {rec} ({lead}), worst-case F1={row['C_full_f1']*100:.2f}% (C_full), first 30 s")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{DOCS}/w3_failure_{rec}.png", dpi=150)
    plt.close(fig)
    print(f"failure plot saved: w3_failure_{rec}.png")
