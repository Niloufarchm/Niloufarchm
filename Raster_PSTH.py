"""
Single-Unit Analysis — Waveforms, Raster, PSTH, ISI, PCA
Fixed version — all bugs from original corrected.

Bug fixes applied
─────────────────
1. Waveform time axis now derived from actual fs (not hardcoded linspace).
2. PSTH x-axis uses bin *centres* instead of left edges.
3. PCA block uses waveforms_ext.get_waveforms_one_unit() — waveforms variable
   was undefined in the original.
4. n_units / n_units_wf are kept separate so the min(6, …) cap for the
   waveform figure does not silently bleed into ISI / PCA loops.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import spikeinterface as si

from neo.io import BlackrockIO

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# SETTINGS
# -----------------------------------------------
FILE_PATH = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/"
    "Monkey/Camelot/electrophysiology/20260330/"
    "cam_20260330_25350_righteyeFF_006.ns6"
)

WF_MS_BEFORE = 1.0   # ms before spike peak
WF_MS_AFTER  = 2.0   # ms after  spike peak

T_BEFORE  = 0.2      # raster/PSTH window (s)
T_AFTER   = 0.2
BIN_SIZE  = 0.01

TARGET_CODE = "64"   # event code marking trial onset

# -----------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------
recording = se.BlackrockRecordingExtractor(FILE_PATH)
fs = recording.get_sampling_frequency()

print("Num channels:", recording.get_num_channels())
print("Channel IDs:", recording.get_channel_ids())
print(f"Sampling rate: {fs} Hz")

# -----------------------------------------------
# 2. PREPROCESS
# -----------------------------------------------
recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
recording.set_dummy_probe_from_locations(np.array([[0.0, 0.0]]))

# -----------------------------------------------
# 3. RUN SORTER
# -----------------------------------------------
sorting = ss.run_sorter(
    sorter_name="tridesclous2",
    recording=recording,
    folder="sorting_output",
    remove_existing_folder=True,
)

unit_ids = sorting.get_unit_ids()
print("Number of units found:", len(unit_ids))
print("Unit IDs:", unit_ids)

if len(unit_ids) == 0:
    raise RuntimeError("No units were found. Nothing to plot.")

# -----------------------------------------------
# 4. CREATE SORTING ANALYZER
# -----------------------------------------------
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="binary_folder",
    folder="analyzer_output",
    sparse=False,
    overwrite=True,
)

analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms", ms_before=WF_MS_BEFORE, ms_after=WF_MS_AFTER)
analyzer.compute("templates", operators=["average", "median", "std"])

print(analyzer)

# -----------------------------------------------
# 5. WAVEFORM PLOT
# FIX: time axis derived from actual fs, not hardcoded linspace length
# -----------------------------------------------
waveforms_ext = analyzer.get_extension("waveforms")

# Correct time axis — number of samples from the real sampling rate
n_samples_wf = int((WF_MS_BEFORE + WF_MS_AFTER) / 1000.0 * fs)
t_wf = np.linspace(-WF_MS_BEFORE, WF_MS_AFTER, n_samples_wf)

# FIX: keep a separate cap for the waveform figure
# so it doesn't overwrite n_units used later
n_units_wf = min(6, len(unit_ids))

fig, axes = plt.subplots(1, n_units_wf, figsize=(3 * n_units_wf, 4))
if n_units_wf == 1:
    axes = [axes]

for i, unit_id in enumerate(unit_ids[:n_units_wf]):
    wf = waveforms_ext.get_waveforms_one_unit(unit_id)   # (n_spikes, n_samples, n_channels)
    wf_ch = wf[:, :, 0]   # single channel

    # Guard: trim time axis to match actual waveform length
    t_plot = t_wf[:wf_ch.shape[1]]

    mean_wf = wf_ch.mean(axis=0)
    std_wf  = wf_ch.std(axis=0)

    for spike in wf_ch[:100]:
        axes[i].plot(t_plot, spike, color="grey", alpha=0.15, linewidth=0.5)

    axes[i].fill_between(t_plot, mean_wf - std_wf, mean_wf + std_wf,
                         color="steelblue", alpha=0.25, linewidth=0)
    axes[i].plot(t_plot, mean_wf, color="steelblue", linewidth=2)
    axes[i].axvline(0, color="red", linestyle="--", linewidth=1,
                    label="spike peak")
    axes[i].set_title(f"Unit {unit_id}\n{len(wf_ch)} spikes")
    axes[i].set_xlabel("Time relative to spike peak (ms)")
    axes[i].set_ylabel("Amplitude (µV)")
    axes[i].spines[["top", "right"]].set_visible(False)
    axes[i].legend(fontsize=7, frameon=False)

plt.suptitle("Spike Waveforms — Mean ± Std + Individual Spikes",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 6. READ BLACKROCK EVENTS
# -----------------------------------------------
base_path = os.path.splitext(FILE_PATH)[0]
reader = BlackrockIO(filename=base_path)
block = reader.read_block()
segment = block.segments[0]

print(f"Found {len(segment.events)} Neo event objects in segment 0")

all_labels_raw = []
for j, ev in enumerate(segment.events):
    labels = np.array(ev.labels).astype(str)
    unique_labels = np.unique(labels)
    print(f"Event object {j}: {len(labels)} events, "
          f"unique labels (first 20) = {unique_labels[:20]}")
    all_labels_raw.extend(labels.tolist())

all_unique_labels = np.unique(np.array(all_labels_raw).astype(str))
print("All unique event labels:", all_unique_labels[:50])

# -----------------------------------------------
# 7. EXTRACT TRIAL ONSETS
# -----------------------------------------------
trial_times_sec = []
for ev in segment.events:
    labels = np.array(ev.labels).astype(str)
    times_sec = ev.times.rescale("s").magnitude
    mask = labels == TARGET_CODE
    if np.any(mask):
        trial_times_sec.extend(times_sec[mask])

trial_times_sec = np.array(sorted(trial_times_sec))
print(f"Found {len(trial_times_sec)} trial onsets with code {TARGET_CODE}")

if len(trial_times_sec) == 0:
    raise RuntimeError(
        f"No events with code {TARGET_CODE} found. "
        "Check that the matching .nev file exists and inspect printed event labels."
    )

# -----------------------------------------------
# 8. RASTER + PSTH FOR ALL UNITS
# FIX: bin centres (not left edges) on PSTH x-axis
# -----------------------------------------------
n_units = len(unit_ids)   # separate from n_units_wf
bins       = np.arange(-T_BEFORE, T_AFTER + BIN_SIZE, BIN_SIZE)
bin_centres = 0.5 * (bins[:-1] + bins[1:])   # FIX: use centres for plotting

fig, axes = plt.subplots(
    n_units, 2,
    figsize=(14, 2.5 * n_units),
    sharex="col",
    squeeze=False,
)

for row, unit_id in enumerate(unit_ids):
    spike_times_sec = sorting.get_unit_spike_train(
        unit_id=unit_id,
        segment_index=0,
        return_times=True,
    )

    aligned_spikes_per_trial = []
    for t0 in trial_times_sec:
        rel_spikes = spike_times_sec - t0
        rel_spikes = rel_spikes[(rel_spikes >= -T_BEFORE) & (rel_spikes <= T_AFTER)]
        aligned_spikes_per_trial.append(rel_spikes)

    # ── Raster ──────────────────────────────────────────────────
    ax_raster = axes[row, 0]
    for trial_idx, rel_spikes in enumerate(aligned_spikes_per_trial):
        ax_raster.scatter(
            rel_spikes,
            np.ones_like(rel_spikes) * trial_idx,
            marker="|", s=35, color="black",
        )
    ax_raster.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_raster.set_ylabel(f"U{unit_id}\nTrial")
    ax_raster.set_title(f"Raster — Unit {unit_id}")
    ax_raster.spines[["top", "right"]].set_visible(False)

    # ── PSTH ────────────────────────────────────────────────────
    ax_psth = axes[row, 1]
    all_aligned = (
        np.concatenate(aligned_spikes_per_trial)
        if any(len(s) for s in aligned_spikes_per_trial)
        else np.array([])
    )

    counts, _ = np.histogram(all_aligned, bins=bins)
    psth_hz   = counts / (len(trial_times_sec) * BIN_SIZE)

    # FIX: plot against bin_centres, not edges[:-1] (left edges)
    ax_psth.fill_between(bin_centres, psth_hz, alpha=0.6)
    ax_psth.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_psth.set_ylabel("Hz")
    ax_psth.set_title(f"PSTH — Unit {unit_id}")
    ax_psth.spines[["top", "right"]].set_visible(False)

axes[-1, 0].set_xlabel(f"Time from event {TARGET_CODE} (s)")
axes[-1, 1].set_xlabel(f"Time from event {TARGET_CODE} (s)")

plt.tight_layout()
plt.show()

# -----------------------------------------------
# 9. ISI HISTOGRAMS
# -----------------------------------------------
fig, axes = plt.subplots(1, n_units, figsize=(3 * n_units, 3))
if n_units == 1:
    axes = [axes]

for i, unit_id in enumerate(unit_ids):   # all units, not capped
    spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
    isi_ms = np.diff(spike_train) / fs * 1000

    violations = np.sum(isi_ms < 3)
    pct = 100 * violations / len(isi_ms) if len(isi_ms) > 0 else 0.0

    axes[i].hist(isi_ms, bins=100, range=(0, 100),
                 color="steelblue", edgecolor="white", linewidth=0.3)
    axes[i].axvline(3, color="red", linestyle="--", linewidth=1.5,
                    label=f"3 ms refractory\nviolations: {pct:.1f}%")
    axes[i].set_title(f"Unit {unit_id}\nISI violations: {pct:.1f}%")
    axes[i].set_xlabel("ISI (ms)")
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=7, frameon=False)
    axes[i].spines[["top", "right"]].set_visible(False)

plt.suptitle("ISI Distributions — Refractory Period Check",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 10. PCA OF SPIKE WAVEFORMS
# FIX: used undefined `waveforms` — now correctly uses waveforms_ext
# -----------------------------------------------
all_waveforms_list = []
all_labels_pca     = []

for unit_id in unit_ids:   # all units
    # FIX: waveforms_ext.get_waveforms_one_unit — not waveforms.get_traces
    wf = waveforms_ext.get_waveforms_one_unit(unit_id)   # (n, samples, ch)
    wf_ch = wf[:, :, 0]   # first channel
    n_take = min(200, len(wf_ch))
    all_waveforms_list.append(wf_ch[:n_take])
    all_labels_pca.extend([unit_id] * n_take)

X = np.vstack(all_waveforms_list)
y = np.array(all_labels_pca)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 6))
for i, unit_id in enumerate(unit_ids):
    mask = y == unit_id
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=f"Unit {unit_id}",
        alpha=0.5, s=15,
        color=plt.cm.tab10(i / max(len(unit_ids), 1)),
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA of Spike Waveforms\n"
             "Well-separated clusters = good unit isolation")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()