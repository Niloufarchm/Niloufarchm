"""
MUA Analysis — Individual Trial Heatmap + Mean Traces
Matches MATLAB MUA code exactly:
  - Shared baseline pooled across ALL trials (not per condition)
  - Two alignments: RF gabor onset + speed change
  - Two conditions: Right+Incongruent (blue) and Right+Congruent (orange)
  - Saves one figure per recording
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

from neo.io import BlackrockIO

# ══════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════
FILE_PATH = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/Monkey/Camelot/electrophysiology/20260408/cam_20260408_28200_MOTIONTASK_SIZE10_Y0X8_008.ns6"
)

# MUA signal
BANDPASS_HZ     = (300, 6000)   # same as MATLAB
DO_RECTIFY      = True          # same as MATLAB do_rectify = true

# Trial window (seconds)
T_BEFORE        = 0.20
T_AFTER         = 0.30

# Baseline for z-score (seconds, relative to each event)
BASELINE        = (-0.20, 0.0)

# Smoothing (Gaussian SD in ms)
SMOOTH_MS       = 5.0

# Latency threshold (z-score)
LATENCY_THRESH  = 1.5

SAVE_DIR        = os.path.dirname(FILE_PATH)

# Condition colours — matches MATLAB cond_colors_rf
COND_COLORS = {
    1: np.array([0.02, 0.45, 0.85]),   # RIGHT + Incongruent  (blue)
    2: np.array([0.93, 0.49, 0.06]),   # RIGHT + Congruent    (orange)
}
COND_LABELS = {
    1: "Incongruent",
    2: "Congruent",
}
PLOT_CONDS  = [1, 2]

# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD + PREPROCESS NS6
# ══════════════════════════════════════════════════════════════
print("Loading NS6 …")
recording = se.BlackrockRecordingExtractor(FILE_PATH)
fs        = recording.get_sampling_frequency()

print(f"  Channels : {recording.get_num_channels()}")
print(f"  fs       : {fs} Hz")
print(f"  Duration : {recording.get_duration():.1f} s")

# Bandpass filter
recording = sp.bandpass_filter(recording,
                                freq_min=BANDPASS_HZ[0],
                                freq_max=BANDPASS_HZ[1])

# ══════════════════════════════════════════════════════════════
#  STEP 2 — PARSE NEV EVENTS  (same as single-unit code)
# ══════════════════════════════════════════════════════════════
print("\nParsing NEV events …")
base_path = os.path.splitext(FILE_PATH)[0]
reader    = BlackrockIO(filename=base_path)
block     = reader.read_block()
segment   = block.segments[0]

all_codes, all_times = [], []
for ev in segment.events:
    times = ev.times.rescale("s").magnitude
    for lbl, t in zip(ev.labels, times):
        try:
            all_codes.append(int(float(str(lbl))))
            all_times.append(float(t))
        except (ValueError, TypeError):
            pass

sort_idx  = np.argsort(all_times)
all_codes = np.array(all_codes, dtype=int)[sort_idx]
all_times = np.array(all_times)[sort_idx]

# Find [128, 128, 64] sequences
seq_positions = [
    i for i in range(2, len(all_codes) - 3)
    if all_codes[i] == 128 and all_codes[i+1] == 128 and all_codes[i+2] == 64
]
print(f"  [128 128 64] sequences : {len(seq_positions)}")

NUM_STIM       = 32
cond_side      = [0]*16 + [1]*16
cond_congruent = ([0]*8  + [1]*8) * 2

trials = []
for ix in seq_positions:
    d1 = all_codes[ix - 2]
    d2 = all_codes[ix - 1]
    if not (1 <= d1 <= 10 and 1 <= d2 <= 10):
        continue
    stim_id = (d1 - 1) * 10 + (d2 - 1) + 1
    if not (1 <= stim_id <= NUM_STIM):
        continue

    side = cond_side[stim_id - 1]
    cong = cond_congruent[stim_id - 1]
    cond_code = (1 - side) * 2 + cong + 1

    t_spd  = all_times[ix + 2]
    right_is_second = (side == 1 and cong == 1) or (side == 0 and cong == 0)
    t_stim = all_times[ix + 1] if right_is_second else all_times[ix]
    t_resp = (all_times[ix + 3] - t_spd) if (ix + 3 < len(all_times)) else None

    trials.append({
        "cond_code": cond_code,
        "t_stim"   : t_stim,
        "t_spd"    : t_spd,
        "t_resp"   : t_resp,
    })

print(f"  Valid trials : {len(trials)}")
for c in PLOT_CONDS:
    n = sum(1 for t in trials if t["cond_code"] == c)
    print(f"    {COND_LABELS[c]} : {n}")

resp_list = [t["t_resp"] for t in trials
             if t["t_resp"] is not None and 0 < t["t_resp"] < 10]
if resp_list:
    rt          = np.array(resp_list)
    resp_median = np.median(rt)
    resp_q1     = np.percentile(rt, 25)
    resp_q3     = np.percentile(rt, 75)
    print(f"  RT median={resp_median*1000:.0f} ms  "
          f"IQR=[{resp_q1*1000:.0f}–{resp_q3*1000:.0f}] ms")
else:
    resp_median = resp_q1 = resp_q3 = None

# ══════════════════════════════════════════════════════════════
#  STEP 3 — EXTRACT MUA SIGNAL PER TRIAL
#  For each trial and each alignment event, cut a window of raw
#  (bandpass-filtered, rectified) signal from the NS6 data.
#  This is the continuous MUA trace — NOT spike-sorted.
# ══════════════════════════════════════════════════════════════
n_samples_trial = int((T_BEFORE + T_AFTER) * fs)
timevec         = np.linspace(-T_BEFORE, T_AFTER, n_samples_trial)
baseline_mask   = (timevec >= BASELINE[0]) & (timevec <= BASELINE[1])

# Smoothing kernel (Gaussian, sigma in samples)
smooth_sigma = (SMOOTH_MS / 1000.0) * fs if SMOOTH_MS > 0 else 0

def extract_trial_mua(recording, event_time_sec, t_before, t_after, fs,
                      do_rectify, smooth_sigma):
    """
    Cut a single trial of MUA signal from the recording.
    Returns a 1-D array of length n_samples_trial.
    """
    start_samp = int((event_time_sec - t_before) * fs)
    end_samp   = start_samp + int((t_before + t_after) * fs)
    total      = recording.get_num_samples()

    # Guard against out-of-bounds
    if start_samp < 0 or end_samp > total:
        return None

    # Get raw trace (bandpass already applied) — shape (n_samples, n_channels)
    trace = recording.get_traces(
        start_frame=start_samp,
        end_frame=end_samp,
        channel_ids=[recording.get_channel_ids()[0]]
    )[:, 0].astype(float)

    # Rectify (absolute value) — same as MATLAB rectify = true
    if do_rectify:
        trace = np.abs(trace)

    # Gaussian smooth
    if smooth_sigma > 0:
        trace = gaussian_filter1d(trace, sigma=smooth_sigma)

    return trace


print("\nExtracting MUA traces …")

# Two alignments — same structure as MATLAB analyses struct
analyses = [
    {"t_key": "t_stim", "title": "RF Gabor Onset",
     "xlabel": "Time from RF gabor onset (s)", "show_resp": False},
    {"t_key": "t_spd",  "title": "Speed Change",
     "xlabel": "Time from speed change (s)",   "show_resp": True},
]

for an in analyses:
    print(f"\n  Alignment: {an['title']}")

    # ── Extract ALL trials (all conditions) to build shared baseline ──
    # This matches MATLAB: baseline_all pools across all trials
    all_trials_signal = []
    trial_cond_list   = []

    for tr in trials:
        if tr["cond_code"] not in PLOT_CONDS:
            continue
        sig = extract_trial_mua(
            recording, tr[an["t_key"]],
            T_BEFORE, T_AFTER, fs, DO_RECTIFY, smooth_sigma
        )
        if sig is not None:
            all_trials_signal.append(sig)
            trial_cond_list.append(tr["cond_code"])

    if len(all_trials_signal) == 0:
        print("  No valid trials — skipping.")
        continue

    all_trials_signal = np.array(all_trials_signal)  # (n_trials, n_samples)
    trial_cond_list   = np.array(trial_cond_list)

    # ── SHARED BASELINE z-score — matches MATLAB exactly ──
    # Pool baseline samples from ALL trials regardless of condition
    baseline_samples = all_trials_signal[:, baseline_mask]  # (n_trials, n_bl_samps)
    mu_bl = baseline_samples.mean()    # single shared mean
    sd_bl = baseline_samples.std()     # single shared SD
    if sd_bl == 0 or np.isnan(sd_bl):
        sd_bl = 1.0

    print(f"    Shared baseline: μ={mu_bl:.4f}  σ={sd_bl:.4f}")

    # Z-score every trial using shared μ and σ
    trials_z = (all_trials_signal - mu_bl) / sd_bl  # (n_trials, n_samples)

    # ════════════════════════════════════════════════════════
    #  FIGURE — one per alignment
    #  Layout:
    #    Left column:  Incongruent  [heatmap / individual trials]
    #    Right column: Congruent    [heatmap / individual trials]
    #    Bottom row:   Mean ± SEM traces for both conditions
    # ════════════════════════════════════════════════════════
    n_cond = len(PLOT_CONDS)

    fig = plt.figure(figsize=(14, 11), facecolor="white")
    file_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    fig.suptitle(
        f"{file_name}  |  MUA  |  {an['title']}\n"
        f"Z-score baseline: shared across all trials  "
        f"(μ={mu_bl:.2f}  σ={sd_bl:.2f}  window {int(BASELINE[0]*1000)}–0 ms)",
        fontsize=10, fontweight="bold", y=0.99,
    )

    # GridSpec: 2 rows — top=heatmaps, bottom=mean traces
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.5, 1.4],
        hspace=0.40,
    )

    # Top row: one heatmap per condition side by side
    top_row = gridspec.GridSpecFromSubplotSpec(
        1, n_cond, subplot_spec=outer[0], wspace=0.08,
    )

    # Shared colour scale across both heatmaps
    vmax = np.percentile(np.abs(trials_z), 97)
    vmin = -vmax * 0.5   # allow moderate negative values to show
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Sort each condition's trials by mean post-stimulus response
    # (strongest responders on top) — makes structure visible
    def sort_trials_by_response(mat):
        post_mask = timevec > 0
        mean_post = mat[:, post_mask].mean(axis=1)
        return mat[np.argsort(-mean_post)]   # descending

    heatmap_axes = []
    for ci, cond in enumerate(PLOT_CONDS):
        mask_c  = trial_cond_list == cond
        mat_c   = trials_z[mask_c]           # (n_trials_c, n_samples)
        mat_c   = sort_trials_by_response(mat_c)
        n_tr_c  = mat_c.shape[0]

        ax = fig.add_subplot(top_row[ci])
        heatmap_axes.append(ax)

        im = ax.imshow(
            mat_c,
            aspect="auto",
            extent=[-T_BEFORE, T_AFTER, n_tr_c, 0],
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        if an["show_resp"] and resp_median is not None:
            ax.axvline(resp_median, color="grey",
                       linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvspan(resp_q1, resp_q3,
                       color="grey", alpha=0.08, linewidth=0)

        ax.set_xlim(-T_BEFORE, T_AFTER)
        ax.set_xlabel(an["xlabel"], fontsize=9)
        ax.set_ylabel("Trial (sorted by response)" if ci == 0 else "",
                      fontsize=9)
        ax.set_title(
            f"{COND_LABELS[cond]}  (n={n_tr_c})",
            fontsize=10, fontweight="bold",
            color=COND_COLORS[cond],
        )
        ax.tick_params(labelsize=8)
        if ci > 0:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar = fig.colorbar(
        im, ax=heatmap_axes,
        orientation="vertical",
        fraction=0.015, pad=0.02,
        label="MUA (z-score)",
    )
    cbar.ax.tick_params(labelsize=8)

    # ── Bottom row: mean ± SEM traces both conditions ────────
    ax_mean = fig.add_subplot(outer[1])

    latency_vals = {}

    for cond in PLOT_CONDS:
        mask_c  = trial_cond_list == cond
        mat_c   = trials_z[mask_c]
        n_tr_c  = mat_c.shape[0]

        if n_tr_c == 0:
            continue

        mu_c  = mat_c.mean(axis=0)
        sem_c = mat_c.std(axis=0) / np.sqrt(n_tr_c)
        col   = COND_COLORS[cond]

        ax_mean.fill_between(
            timevec, mu_c - sem_c, mu_c + sem_c,
            color=col, alpha=0.20, linewidth=0,
        )
        ax_mean.plot(
            timevec, mu_c,
            color=col, linewidth=2.4,
            label=f"{COND_LABELS[cond]}  (n={n_tr_c})",
        )

        # Latency detection
        post = timevec > 0
        above = np.where(post & (mu_c > LATENCY_THRESH))[0]
        if len(above):
            latency_vals[cond] = timevec[above[0]]
            ax_mean.axvline(
                latency_vals[cond],
                color=col, linewidth=1.8, alpha=0.85,
            )

    ax_mean.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_mean.axhline(0, color="grey",  linestyle=":",  linewidth=0.8)

    # Response time band
    if an["show_resp"] and resp_median is not None:
        ax_mean.axvspan(resp_q1, resp_q3,
                        color="grey", alpha=0.10, linewidth=0)
        ax_mean.axvline(resp_median, color="grey", linestyle="--",
                        linewidth=1.4,
                        label=f"RT median {resp_median*1000:.0f} ms")

    # Latency bracket
    if len(latency_vals) == 2:
        lats   = sorted(latency_vals.values())
        y_vals = [ax_mean.get_ylim()[1] * 0.85] * 2
        ax_mean.annotate(
            "", xy=(lats[1], y_vals[0]), xytext=(lats[0], y_vals[0]),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
        )
        ax_mean.text(
            np.mean(lats), y_vals[0] * 1.05,
            f"Δ = {(lats[1]-lats[0])*1000:.0f} ms",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )

    ax_mean.set_xlim(-T_BEFORE, T_AFTER)
    ax_mean.set_xlabel(an["xlabel"], fontsize=10)
    ax_mean.set_ylabel("MUA (z-score)", fontsize=10)
    #ax_mean.set_title(
    #   "Mean ± SEM — Right visual field change\n"
    #    "Shared z-score baseline (all trials, all conditions)",
    #   fontsize=9, fontweight="bold",
    #)
    ax_mean.legend(fontsize=9, frameon=False, loc="upper left")
    ax_mean.spines[["top", "right"]].set_visible(False)
    ax_mean.tick_params(labelsize=9)

    # ── Save ─────────────────────────────────────────────────
    out = os.path.join(
        SAVE_DIR,
        f"{file_name}_MUA_{an['title'].replace(' ', '_')}.png"
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.show()
    plt.close(fig)

print("\nDone.")