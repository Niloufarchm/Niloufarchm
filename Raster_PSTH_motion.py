"""
vertical line for the motion onset figure based on; PSTH bin after t=0 where the z-score exceeds 1.5
Single Unit Analysis — Trial-aligned Raster + PSTH
Preprocessing matches the working approach: bandpass only, no common reference.

Waveform window: ms_before=1.0, ms_after=2.0
    0 = spike peak (sorter detection point)
   -1 ms to 0: rise before peak
    0 to +2 ms: after-hyperpolarisation (AHP)

Conditions (same as MATLAB MUA code):
    Code 1: RIGHT change + Incongruent  (RF gabor appears FIRST)
    Code 2: RIGHT change + Congruent    (RF gabor appears SECOND)
    Code 3: LEFT  change + Incongruent
    Code 4: LEFT  change + Congruent

Two alignments per unit:
    A) Right RF gabor onset
    B) Speed change onset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import spikeinterface as si

from neo.io import BlackrockIO

# ══════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════
FILE_PATH  = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/Monkey/Camelot/electrophysiology/20260408/cam_20260408_28200_MOTIONTASK_SIZE10_Y0X8_008.ns6"
)

T_BEFORE        = 0.20      # seconds before event
T_AFTER         = 0.30      # seconds after event
BIN_SIZE        = 0.010     # PSTH bin size in seconds
SMOOTH_MS       = 5.0       # Gaussian smoothing SD in ms (0 = off)
BASELINE        = (-0.20, 0.0)   # baseline window for z-score
LATENCY_THRESH  = 1.5       # z-score threshold for latency detection

# Waveform extraction window
WF_MS_BEFORE    = 1.0       # ms before spike peak
WF_MS_AFTER     = 2.0       # ms after spike peak

SAVE_DIR        = os.path.dirname(FILE_PATH)

# Condition colors — blue (incongruent) and orange (congruent)
COND_COLORS = {
    1: np.array([0.02, 0.45, 0.85]),
    2: np.array([0.93, 0.49, 0.06]),
    3: np.array([0.40, 0.76, 0.35]),
    4: np.array([0.75, 0.22, 0.17]),
}
COND_LABELS = {
    1: "Right+Incongruent",
    2: "Right+Congruent",
    3: "Left+Incongruent",
    4: "Left+Congruent",
}
PLOT_CONDS = [1, 2]   # only RIGHT-field change conditions

# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD + PREPROCESS
# ══════════════════════════════════════════════════════════════
print("Loading NS6 …")
recording = se.BlackrockRecordingExtractor(FILE_PATH)
fs        = recording.get_sampling_frequency()

print(f"  Channels : {recording.get_num_channels()}")
print(f"  fs       : {fs} Hz")
print(f"  Duration : {recording.get_duration():.1f} s")

# Bandpass filter
recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)

# Dummy probe location — required by tridesclous2 for single electrode
recording.set_dummy_probe_from_locations(np.array([[0.0, 0.0]]))

print(f"  Channel locations: {recording.get_channel_locations()}")

# ══════════════════════════════════════════════════════════════
#  STEP 2 — SPIKE SORTING
# ══════════════════════════════════════════════════════════════
print("\nRunning spike sorter …")
sorting = ss.run_sorter(
    sorter_name="tridesclous2",
    recording=recording,
    folder="sorting_output",
    remove_existing_folder=True,
)

unit_ids = sorting.get_unit_ids()
print(f"  Units found : {len(unit_ids)}")
print(f"  Unit IDs   : {unit_ids}")

if len(unit_ids) == 0:
    raise RuntimeError("No units found. Nothing to plot.")

# ══════════════════════════════════════════════════════════════
#  STEP 3 — SORTING ANALYZER + WAVEFORMS
# ══════════════════════════════════════════════════════════════
print("\nExtracting waveforms …")
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="binary_folder",
    folder="analyzer_output",
    sparse=False,
    overwrite=True,
)
analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms",  ms_before=WF_MS_BEFORE, ms_after=WF_MS_AFTER)
analyzer.compute("templates",  operators=["average", "median", "std"])
print(analyzer)

waveforms_ext = analyzer.get_extension("waveforms")

# Correct time axis for waveforms — derived from actual sampling rate
# n_samples = ms_before + ms_after converted to samples
n_samples_wf = int((WF_MS_BEFORE + WF_MS_AFTER) / 1000.0 * fs)
t_wf = np.linspace(-WF_MS_BEFORE, WF_MS_AFTER, n_samples_wf)

# ══════════════════════════════════════════════════════════════
#  STEP 4 — PARSE NEV EVENTS
#  Logic identical to MATLAB: find [128, 128, 64] sequences
# ══════════════════════════════════════════════════════════════
print("\nParsing NEV events …")
base_path = os.path.splitext(FILE_PATH)[0]
reader    = BlackrockIO(filename=base_path)
block     = reader.read_block()
segment   = block.segments[0]

print(f"  Event objects in segment 0 : {len(segment.events)}")

# Inspect labels (same as working code)
all_labels_raw = []
for j, ev in enumerate(segment.events):
    labels = np.array(ev.labels).astype(str)
    unique = np.unique(labels)
    print(f"  Event object {j}: {len(labels)} events, "
          f"unique labels (first 20) = {unique[:20]}")
    all_labels_raw.extend(labels.tolist())

all_unique = np.unique(np.array(all_labels_raw).astype(str))
print("  All unique labels:", all_unique[:50])

# Build sorted arrays of all codes + times
all_codes = []
all_times = []
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

print(f"  Total numeric events : {len(all_codes)}")

# ──────────────────────────────────────────────────────────────
#  Find [128, 128, 64] sequences
# ──────────────────────────────────────────────────────────────
seq_positions = [
    i for i in range(2, len(all_codes) - 3)
    if (all_codes[i]   == 128 and
        all_codes[i+1] == 128 and
        all_codes[i+2] == 64)
]
print(f"  [128 128 64] sequences : {len(seq_positions)}")

# ──────────────────────────────────────────────────────────────
#  32-stimulus condition arrays
# ──────────────────────────────────────────────────────────────
NUM_STIM       = 32
cond_side      = [0]*16 + [1]*16
cond_congruent = ([0]*8 + [1]*8) * 2

# ──────────────────────────────────────────────────────────────
#  Extract trial parameters
# ──────────────────────────────────────────────────────────────
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

    t_spd = all_times[ix + 2]

    right_is_second = (side == 1 and cong == 1) or (side == 0 and cong == 0)
    t_stim = all_times[ix + 1] if right_is_second else all_times[ix]

    t_resp = (all_times[ix + 3] - t_spd) if (ix + 3 < len(all_times)) else None

    trials.append({
        "cond_code": cond_code,
        "t_stim"   : t_stim,
        "t_spd"    : t_spd,
        "t_resp"   : t_resp,
    })

print(f"  Valid trials parsed : {len(trials)}")
for c in [1, 2, 3, 4]:
    n = sum(1 for t in trials if t["cond_code"] == c)
    print(f"    {COND_LABELS[c]} : {n}")

# Response time summary
resp_list = [t["t_resp"] for t in trials
             if t["t_resp"] is not None and 0 < t["t_resp"] < 10]
if resp_list:
    rt = np.array(resp_list)
    resp_median = np.median(rt)
    resp_q1     = np.percentile(rt, 25)
    resp_q3     = np.percentile(rt, 75)
    print(f"\n  Response time  median={resp_median*1000:.0f} ms  "
          f"IQR=[{resp_q1*1000:.0f}–{resp_q3*1000:.0f}] ms  (n={len(rt)})")
else:
    resp_median = resp_q1 = resp_q3 = None

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def gaussian_smooth(arr, smooth_ms, bin_size_s):
    """Smooth a 1-D array with a Gaussian kernel."""
    if smooth_ms <= 0:
        return arr
    sigma = (smooth_ms / 1000.0) / bin_size_s
    hw    = int(4 * sigma)
    x     = np.arange(-hw, hw + 1)
    k     = np.exp(-0.5 * (x / sigma) ** 2)
    k    /= k.sum()
    return np.convolve(arr, k, mode="same")


def align_spikes(spike_times, event_times, t_before, t_after):
    """Return list of relative-spike-time arrays, one per trial."""
    result = []
    for t0 in event_times:
        rel = spike_times - t0
        result.append(rel[(rel >= -t_before) & (rel <= t_after)])
    return result


def psth_zscore(aligned, bin_edges, baseline_mask, bin_size):
    """
    Compute z-scored PSTH and SEM.
    Z-score baseline is defined by baseline_mask over bin_centres.
    """
    n_tr   = len(aligned)
    n_bins = len(bin_edges) - 1

    if n_tr == 0:
        z = np.zeros(n_bins)
        return z, z

    mat = np.zeros((n_tr, n_bins))
    for i, spks in enumerate(aligned):
        counts, _ = np.histogram(spks, bins=bin_edges)
        mat[i]    = counts / bin_size   # convert to Hz

    mu_raw  = mat.mean(axis=0)
    sem_raw = mat.std(axis=0) / np.sqrt(n_tr)

    mu_bl = mu_raw[baseline_mask].mean()
    sd_bl = mu_raw[baseline_mask].std()
    if sd_bl == 0 or np.isnan(sd_bl):
        sd_bl = 1.0

    return (mu_raw - mu_bl) / sd_bl, sem_raw / sd_bl


def waveform_width_ms(mean_wf, t_axis):
    """
    Estimate spike width at half-maximum of the trough.
    Returns width in ms, or NaN if not found.
    This is a simple proxy for broad (pyramidal) vs narrow (interneuron).
    """
    trough_idx = np.argmin(mean_wf)
    half_amp   = mean_wf[trough_idx] / 2.0
    left  = np.where(mean_wf[:trough_idx] > half_amp)[0]
    right = np.where(mean_wf[trough_idx:] > half_amp)[0]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    left_idx  = left[-1]
    right_idx = trough_idx + right[0]
    return t_axis[right_idx] - t_axis[left_idx]


# ══════════════════════════════════════════════════════════════
#  STEP 5 — PLOT EACH UNIT
# ══════════════════════════════════════════════════════════════
bin_edges     = np.arange(-T_BEFORE, T_AFTER + BIN_SIZE, BIN_SIZE)
bin_centres   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
baseline_mask = (bin_centres >= BASELINE[0]) & (bin_centres <= BASELINE[1])

file_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

analyses = [
    {"t_key": "t_stim", "title": "RF Gabor Onset",
     "xlabel": "Time from RF gabor onset (s)", "show_resp": False},
    {"t_key": "t_spd",  "title": "Speed Change",
     "xlabel": "Time from speed change (s)",   "show_resp": True},
]

for unit_id in unit_ids:

    spike_times = sorting.get_unit_spike_train(
        unit_id=unit_id, segment_index=0, return_times=True
    )
    n_spikes = len(spike_times)

    if n_spikes < 10:
        print(f"\nUnit {unit_id}: only {n_spikes} spikes — skipping.")
        continue

    print(f"\nPlotting unit {unit_id}  ({n_spikes} spikes) …")

    # ── Figure ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13), facecolor="white")
    fig.suptitle(
        f"{file_name}  |  Unit {unit_id}  |  {n_spikes} spikes",
        fontsize=11, fontweight="bold", y=0.99,
    )

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 2.2, 2.2],
        hspace=0.50,
    )

    # ── Row 0: Waveform + ISI ────────────────────────────────
    r0     = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.40)
    ax_wf  = fig.add_subplot(r0[0])
    ax_isi = fig.add_subplot(r0[1])

    # ── Waveform ─────────────────────────────────────────────
    wf    = waveforms_ext.get_waveforms_one_unit(unit_id)  # (n, samples, ch)
    wf_ch = wf[:, :, 0]   # single channel
    mean_wf = wf_ch.mean(axis=0)
    std_wf  = wf_ch.std(axis=0)

    # Individual spikes in grey
    for spike in wf_ch[:100]:
        ax_wf.plot(t_wf, spike, color="grey", alpha=0.15, linewidth=0.5)

    # Mean ± std shading
    ax_wf.fill_between(
        t_wf,
        mean_wf - std_wf,
        mean_wf + std_wf,
        color="steelblue", alpha=0.25, linewidth=0,
    )
    ax_wf.plot(t_wf, mean_wf, color="steelblue", linewidth=2.2)

    # Spike peak marker at t = 0
    ax_wf.axvline(0, color="red", linestyle="--", linewidth=1.0,
                  label="spike peak")

    # Spike width annotation
    width_ms = waveform_width_ms(mean_wf, t_wf)
    if not np.isnan(width_ms):
        ax_wf.text(
            0.97, 0.95,
            f"width ≈ {width_ms:.2f} ms",
            transform=ax_wf.transAxes,
            ha="right", va="top", fontsize=7,
            color="steelblue",
        )

    ax_wf.set_xlabel("Time relative to spike peak (ms)", fontsize=9)
    ax_wf.set_ylabel("Amplitude (µV)", fontsize=9)
    ax_wf.set_title("Spike Waveform\n"
                    "(grey = individual spikes,  blue = mean ± std)",
                    fontsize=9, fontweight="bold")
    ax_wf.legend(fontsize=7, frameon=False)
    ax_wf.spines[["top", "right"]].set_visible(False)
    ax_wf.tick_params(labelsize=8)

    # Add x-axis region labels
    ax_wf.text(-WF_MS_BEFORE * 0.95, ax_wf.get_ylim()[0],
               "← rise", ha="left", va="bottom",
               fontsize=6, color="grey", style="italic")
    ax_wf.text(WF_MS_AFTER * 0.95, ax_wf.get_ylim()[0],
               "AHP →", ha="right", va="bottom",
               fontsize=6, color="grey", style="italic")

    # ── ISI ──────────────────────────────────────────────────
    isi_ms   = np.diff(spike_times) * 1000
    viol     = np.sum(isi_ms < 3.0)
    pct_viol = 100.0 * viol / len(isi_ms) if len(isi_ms) > 0 else 0.0
    mean_fr  = n_spikes / recording.get_duration()

    ax_isi.hist(isi_ms, bins=100, range=(0, 100),
                color="steelblue", edgecolor="white", linewidth=0.3)
    ax_isi.axvline(3, color="red", linestyle="--", linewidth=1.5,
                   label=f"3 ms refractory\nviolations: {pct_viol:.1f}%")
    ax_isi.set_xlabel("ISI (ms)", fontsize=9)
    ax_isi.set_ylabel("Count", fontsize=9)
    ax_isi.set_title(
        f"ISI Distribution\nmean FR = {mean_fr:.1f} Hz",
        fontsize=9, fontweight="bold"
    )
    ax_isi.legend(fontsize=8, frameon=False)
    ax_isi.spines[["top", "right"]].set_visible(False)
    ax_isi.tick_params(labelsize=8)

    # ── Rows 1 & 2: Alignments ──────────────────────────────
    for a_idx, an in enumerate(analyses):

        r         = gridspec.GridSpecFromSubplotSpec(
                        1, 2, subplot_spec=outer[a_idx + 1], wspace=0.32)
        ax_raster = fig.add_subplot(r[0])
        ax_psth   = fig.add_subplot(r[1])

        # Collect per-condition data
        psth_store   = {}
        raster_store = {}

        for cond in PLOT_CONDS:
            t_events = np.array([
                t[an["t_key"]] for t in trials if t["cond_code"] == cond
            ])
            if len(t_events) == 0:
                continue

            aligned        = align_spikes(spike_times, t_events, T_BEFORE, T_AFTER)
            psth_z, sem_z  = psth_zscore(aligned, bin_edges, baseline_mask, BIN_SIZE)
            psth_sm        = gaussian_smooth(psth_z, SMOOTH_MS, BIN_SIZE)
            sem_sm         = gaussian_smooth(sem_z,  SMOOTH_MS, BIN_SIZE)

            psth_store[cond]   = (psth_sm, sem_sm, len(t_events))
            raster_store[cond] = aligned

        # ── Raster ──────────────────────────────────────────
        y_offset  = 0
        dividers  = []

        for cond in PLOT_CONDS:
            if cond not in raster_store:
                continue
            col     = COND_COLORS[cond]
            aligned = raster_store[cond]
            n_tr    = len(aligned)

            for k, rel_spikes in enumerate(aligned):
                ax_raster.scatter(
                    rel_spikes,
                    np.full(len(rel_spikes), y_offset + k),
                    marker="|", s=30, linewidths=0.8, color=col,
                )

            # Condition label in left margin
            ax_raster.text(
                -T_BEFORE * 1.03, y_offset + n_tr / 2,
                COND_LABELS[cond].split("+")[1],
                ha="right", va="center", fontsize=7,
                color=col, fontweight="bold",
            )
            dividers.append(y_offset + n_tr)
            y_offset += n_tr

        # Divider line between conditions
        if len(dividers) >= 1:
            ax_raster.axhline(dividers[0] - 0.5,
                              color="grey", linewidth=0.6, linestyle=":")

        ax_raster.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax_raster.set_xlim(-T_BEFORE, T_AFTER)
        ax_raster.set_ylim(-0.5, max(y_offset - 0.5, 0.5))
        ax_raster.set_xlabel(an["xlabel"], fontsize=9)
        ax_raster.set_ylabel("Trial", fontsize=9)
        ax_raster.set_title(
            f"Raster — {an['title']}",
            fontsize=10, fontweight="bold"
        )
        ax_raster.spines[["top", "right"]].set_visible(False)
        ax_raster.tick_params(labelsize=8)

        # ── PSTH ────────────────────────────────────────────
        all_vals = []

        for cond in PLOT_CONDS:
            if cond not in psth_store:
                continue
            psth_z, sem_z, n_tr = psth_store[cond]
            col = COND_COLORS[cond]

            ax_psth.fill_between(
                bin_centres, psth_z - sem_z, psth_z + sem_z,
                color=col, alpha=0.20, linewidth=0,
            )
            ax_psth.plot(
                bin_centres, psth_z,
                color=col, linewidth=2.2,
                label=f"{COND_LABELS[cond].split('+')[1]}  (n={n_tr})",
            )
            all_vals.extend((psth_z + sem_z).tolist())
            all_vals.extend((psth_z - sem_z).tolist())

        ax_psth.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax_psth.axhline(0, color="grey",  linestyle=":",  linewidth=0.8)

        # Response time + latency annotation (speed-change alignment only)
        if an["show_resp"] and resp_median is not None:
            ax_psth.axvspan(resp_q1, resp_q3,
                            color="grey", alpha=0.10, linewidth=0)
            ax_psth.axvline(
                resp_median, color="grey", linestyle="--", linewidth=1.4,
                label=f"RT median {resp_median*1000:.0f} ms",
            )

            # Latency detection per condition
            post     = bin_centres > 0
            lat_vals = {}
            for cond in PLOT_CONDS:
                if cond not in psth_store:
                    continue
                p_z   = psth_store[cond][0]
                above = np.where(post & (p_z > LATENCY_THRESH))[0]
                if len(above):
                    lat_vals[cond] = bin_centres[above[0]]
                    ax_psth.axvline(
                        lat_vals[cond],
                        color=COND_COLORS[cond],
                        linewidth=1.8, alpha=0.85,
                    )

            # Latency bracket
            if len(lat_vals) == 2:
                lats  = sorted(lat_vals.values())
                y_top = max(all_vals) * 0.88 if all_vals else 1
                span  = (max(all_vals) - min(all_vals)) * 0.04 if all_vals else 0.1
                ax_psth.annotate(
                    "", xy=(lats[1], y_top), xytext=(lats[0], y_top),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
                )
                ax_psth.text(
                    np.mean(lats), y_top + span * 1.5,
                    f"Δ = {(lats[1]-lats[0])*1000:.0f} ms",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

        ax_psth.set_xlim(-T_BEFORE, T_AFTER)
        ax_psth.set_xlabel(an["xlabel"], fontsize=9)
        ax_psth.set_ylabel("Firing rate (z-score)", fontsize=9)
        ax_psth.set_title(
            f"PSTH — {an['title']}",
            fontsize=10, fontweight="bold"
        )
        ax_psth.legend(fontsize=8, frameon=False, loc="upper left")
        ax_psth.spines[["top", "right"]].set_visible(False)
        ax_psth.tick_params(labelsize=8)

    # ── Save + show ──────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(SAVE_DIR, f"{file_name}_unit{unit_id}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.show()
    plt.close(fig)

print("\nAll units done.")