"""ECG processing library: baseline removal, powerline notch, Pan-Tompkins QRS detector, beat matching evaluation."""
import numpy as np
from scipy import ndimage, signal

BEAT_SYMBOLS = set("NLRBAaJSVrejFnEQ?/")


def remove_baseline(sig, fs):
    w1 = int(0.2 * fs) | 1
    w2 = int(0.6 * fs) | 1
    baseline = signal.medfilt(sig, w1)
    baseline = signal.medfilt(baseline, w2)
    return sig - baseline


def notch_powerline(sig, fs, f0=60.0, q=35.0):
    b, a = signal.iirnotch(f0, q, fs)
    return signal.filtfilt(b, a, sig)


def bandpass_pt(sig, fs, low=5.0, high=15.0, order=1):
    b, a = signal.butter(order, [low, high], btype="band", fs=fs)
    return signal.lfilter(b, a, sig)


def mwa_past(x, win):
    ret = np.cumsum(x, dtype=float)
    ret[win:] = ret[win:] - ret[:-win]
    for i in range(1, win):
        ret[i - 1] = ret[i - 1] / i
    ret[win - 1:] = ret[win - 1:] / win
    return ret


def _refine_on_signal(sig_proc, idx_list, win):
    back = int(win * 1.5)
    fwd = int(win * 0.25)
    refined = []
    for p in idx_list:
        lo = max(0, p - back)
        hi = min(len(sig_proc), p + fwd + 1)
        refined.append(lo + int(np.argmax(sig_proc[lo:hi])))
    return np.asarray(refined, dtype=int)


def _refine_on_slope(derivative, idx_list, win):
    back = int(win * 1.5)
    fwd = int(win * 0.25)
    refined = []
    for p in idx_list:
        lo = max(0, p - back)
        hi = min(len(derivative), p + fwd + 1)
        refined.append(lo + int(np.argmax(derivative[lo:hi])))
    return np.asarray(refined, dtype=int)


def pan_tompkins(sig, fs, return_stages=False):
    filtered = bandpass_pt(sig, fs)
    derivative = np.diff(filtered)
    squared = derivative ** 2
    win = int(0.15 * fs)
    integrated = mwa_past(squared.astype(float), win)
    integrated[: int(0.6 * fs)] = 0.0

    peaks, _ = signal.find_peaks(integrated)
    spki = 0.0
    npki = 0.0
    refractory = int(0.3 * fs)
    mean_rr = int(0.75 * fs)
    recent_rr = []
    last_qrs = None
    qrs_list = []

    def accept(p, peak_val, strong):
        nonlocal spki, last_qrs
        if strong:
            spki = 0.125 * peak_val + 0.875 * spki
        else:
            spki = 0.25 * peak_val + 0.75 * spki
        if last_qrs is not None:
            rr = p - last_qrs
            recent_rr.append(rr)
            recent_rr[:] = recent_rr[-8:]
            mean_rr = int(np.median(recent_rr))
        last_qrs = p
        qrs_list.append(p)

    for p in peaks:
        v = integrated[p]
        thr1 = npki + 0.25 * (spki - npki)
        thr2 = 0.5 * thr1
        if v > thr1:
            if last_qrs is not None and p - last_qrs < refractory:
                continue
            if last_qrs is not None and len(recent_rr) >= 8 and p - last_qrs > 1.66 * mean_rr:
                gap = peaks[(peaks > last_qrs + refractory) & (peaks < p)]
                gap = [g for g in gap if integrated[g] > thr2]
                if gap:
                    g = max(gap, key=lambda x: integrated[x])
                    accept(g, integrated[g], strong=False)
            accept(p, v, strong=True)
        else:
            npki = 0.875 * npki + 0.125 * v

    qrs = _refine_on_slope(derivative, qrs_list, win)
    if return_stages:
        stages = dict(filtered=filtered, derivative=derivative, squared=squared,
                      integrated=integrated, win=win, qrs_raw=qrs_list)
        return qrs, stages
    return qrs


def evaluate(detected, reference, tolerance_samples):
    det = np.sort(np.asarray(detected, dtype=int))
    ref = np.asarray(reference, dtype=int)
    matched = np.zeros(len(det), dtype=bool)
    tp = 0
    for r in ref:
        cand = np.where((np.abs(det - r) <= tolerance_samples) & ~matched)[0]
        if cand.size:
            j = cand[np.argmin(np.abs(det[cand] - r))]
            matched[j] = True
            tp += 1
    fp = int((~matched).sum())
    fn = len(ref) - tp
    se = tp / (tp + fn) if (tp + fn) else 1.0
    pp = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * se * pp / (se + pp) if (se + pp) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, se=se, pp=pp, f1=f1)
