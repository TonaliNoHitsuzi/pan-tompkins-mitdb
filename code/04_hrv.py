"""W3b: heart rate and HRV analysis on record 100, detected vs reference R peaks."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
from scipy import signal as sps
from ecg_lib import BEAT_SYMBOLS, remove_baseline, notch_powerline, pan_tompkins

DATA_DIR = r"data/mit-bih-arrhythmia-database-1.0.0"
DOCS = r"docs"
FS = 360
REC = "100"

record = wfdb.rdrecord(f"{DATA_DIR}/{REC}")
ann = wfdb.rdann(f"{DATA_DIR}/{REC}", extension="atr")
sig = record.p_signal[:, record.sig_name.index("MLII")]
ref = np.sort(np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS]))
det = pan_tompkins(remove_baseline(notch_powerline(sig, FS, 60.0), FS), FS)


def rr_series(peaks, fs):
    rr = np.diff(peaks) / fs
    keep = (rr > 0.3) & (rr < 2.0)
    return rr[keep], peaks[1:][keep]


def hrv_time(rr_ms):
    d = np.diff(rr_ms)
    sdnn = float(np.std(rr_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    pnn50 = float(np.mean(np.abs(d) > 50) * 100)
    mean_hr = float(60000 / np.mean(rr_ms))
    return dict(MeanHR_bpm=mean_hr, SDNN_ms=sdnn, RMSSD_ms=rmssd, pNN50_pct=pnn50)


def hrv_freq(rr, fs_rr=4.0):
    t = np.arange(0, len(rr) / fs_rr, 1 / fs_rr)[: len(rr)]
    rr_interp = np.interp(np.arange(0, t[-1], 1 / fs_rr), np.cumsum(rr), rr)
    rr_interp = rr_interp - rr_interp.mean()
    f, p = sps.welch(rr_interp, fs=fs_rr, nperseg=256)
    lf = float(np.trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
    hf = float(np.trapezoid(p[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)]))
    return dict(LF_ms2=lf, HF_ms2=hf, LF_HF=lf / hf if hf else float("nan"))


rr_det, pk_det = rr_series(det, FS)
rr_ref, pk_ref = rr_series(ref, FS)

time_det, time_ref = hrv_time(rr_det * 1000), hrv_time(rr_ref * 1000)
freq_det, freq_ref = hrv_freq(rr_det), hrv_freq(rr_ref)

print(f"{'metric':14s} {'detected':>12s} {'reference':>12s} {'rel.err%':>10s}")
for k in time_det:
    e = abs(time_det[k] - time_ref[k]) / time_ref[k] * 100
    print(f"{k:14s} {time_det[k]:12.3f} {time_ref[k]:12.3f} {e:10.2f}")
for k in freq_det:
    e = abs(freq_det[k] - freq_ref[k]) / freq_ref[k] * 100
    print(f"{k:14s} {freq_det[k]:12.3e} {freq_ref[k]:12.3e} {e:10.2f}")

hr_det = 60 / rr_det
hr_ref = 60 / rr_ref
t_det = pk_det / FS / 60
t_ref = pk_ref / FS / 60

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(t_ref, rr_ref * 1000, linewidth=0.8, color="tab:blue", label="reference")
axes[0].plot(t_det, rr_det * 1000, linewidth=0.8, color="tab:red", alpha=0.7, label="detected")
axes[0].set_ylabel("RR interval (ms)")
axes[0].set_title(f"record {REC}: RR tachogram, detected vs reference")
axes[1].plot(t_ref, hr_ref, linewidth=0.8, color="tab:blue", label="reference")
axes[1].plot(t_det, hr_det, linewidth=0.8, color="tab:red", alpha=0.7, label="detected")
axes[1].set_ylabel("instantaneous HR (bpm)")
axes[1].set_xlabel("time (min)")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DOCS}/w3_hrv_{REC}.png", dpi=150)

f, p = sps.welch(np.interp(np.arange(0, np.cumsum(rr_det)[-1], 1 / 4), np.cumsum(rr_det), rr_det - rr_det.mean()), fs=4, nperseg=256)
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(f, p, linewidth=0.9)
ax.axvspan(0.04, 0.15, alpha=0.15, color="orange", label="LF")
ax.axvspan(0.15, 0.4, alpha=0.15, color="green", label="HF")
ax.set_xlim(0, 0.5)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (s^2/Hz)")
ax.set_title(f"record {REC}: RR interval spectrum (detected peaks)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DOCS}/w3_hrv_spectrum_{REC}.png", dpi=150)
print(f"plots saved: w3_hrv_{REC}.png, w3_hrv_spectrum_{REC}.png")
