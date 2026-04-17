"""
LGN Cell-Type Classification Pipeline
======================================
Loops over every session folder directly under ELECTROPHYSIOLOGY_ROOT,
runs spike sorting (or loads cached results), extracts per-unit features,
then clusters all units with GMM.

Folder layout expected
──────────────────────
electrophysiology/
    20260330/   ← session folders  (one level deep, scanned here)
        *.ns6
    20260408/
        *.ns6
    ...

Output
──────
    results/
        features.csv          ← one row per unit, all features
        cluster_summary.csv   ← cluster means + std
        pca_overview.png      ← PCA coloured by cluster + session
        umap_overview.png     ← UMAP
        bic_curve.png         ← BIC for k = 1..MAX_K
        per_unit_waveforms/
            <session>_unit<id>.png  ← waveform + ISI for each unit

Features extracted
──────────────────
  Waveform
    wf_width_ms          half-width of the trough (narrow=interneuron/M, broad=P)
    wf_peak_to_trough    depolarisation peak / trough amplitude ratio
    wf_ahp_depth         amplitude at fixed post-spike window (AHP proxy)
    wf_repol_slope       mean slope from trough to AHP
    wf_trough_asymmetry  area before / area after trough
    wf_pc1..wf_pc5       first 5 PCs of the mean waveform

  Firing statistics
    fr_mean_hz           overall mean firing rate
    isi_cv               CV of inter-spike intervals  (regularity)
    isi_burst_pct        % of ISIs < 6 ms             (burstiness)
    isi_mode_ms          mode of ISI distribution
    isi_viol_pct         % ISIs < 3 ms                (unit quality)
    fano_factor          Fano factor in 100 ms windows

  Metadata
    session              folder name
    depth_idx            recording depth hint from filename (if present)
    unit_id              sorter unit id
    n_spikes             total spike count
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves files instead of showing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spikeinterface.extractors   as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters      as ss
import spikeinterface              as si

from sklearn.decomposition        import PCA
from sklearn.preprocessing        import StandardScaler
from sklearn.mixture              import GaussianMixture
from sklearn.metrics              import silhouette_score
from scipy.stats                  import mode as scipy_mode

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
ELECTROPHYSIOLOGY_ROOT = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/"
    "Monkey/Camelot/electrophysiology/"
)

# Keep results next to the data but cache LOCALLY to avoid macOS ghost files.
# The external SSD (ExFAT / non-APFS) causes Finder to create hidden  ._*
# metadata companions for every .npy file. SpikeInterface then tries to
# unpickle those ghost files and crashes. A local cache path avoids this.
RESULTS_DIR  = os.path.join(ELECTROPHYSIOLOGY_ROOT, "..", "ml_results")
CACHE_DIR    = os.path.expanduser("~/sorting_cache_camelot")   # LOCAL — no ghost files

WF_MS_BEFORE = 1.0     # ms before spike peak
WF_MS_AFTER  = 2.0     # ms after  spike peak
MIN_SPIKES   = 50      # skip units with fewer spikes
MAX_SPIKES_WF = 500    # waveforms sampled per unit (sorter default)
MAX_WF_PLOT  = 80      # individual waveforms drawn in per-unit plots
FANO_WIN_S   = 0.100   # window for Fano factor (s)

MAX_K        = 8       # maximum number of GMM clusters to try
RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def find_session_folders(root: str) -> list:
    """Return sorted list of direct child directories that contain ≥1 .ns6 file."""
    sessions = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        ns6_files = [f for f in os.listdir(path) if f.endswith(".ns6")]
        if ns6_files:
            sessions.append((name, path, ns6_files))
    return sessions


def extract_depth_from_filename(filename: str) -> float:
    """
    Try to parse a depth value (µm) from filename conventions like:
        cam_20260330_25350_*.ns6   → 25350 µm (second numeric field)
    Returns NaN if not found.
    """
    parts = re.split(r"[_.]", os.path.basename(filename))
    numbers = [p for p in parts if p.isdigit() and len(p) >= 4]
    if len(numbers) >= 2:
        try:
            return float(numbers[1])
        except ValueError:
            pass
    return float("nan")


def purge_macos_ghost_files(directory: str) -> int:
    """
    Recursively delete macOS '._*' AppleDouble metadata companions.
    These are created by Finder on ExFAT / non-APFS volumes and are NOT
    valid numpy/pickle files — SpikeInterface trips over them on load.
    Returns the number of files removed.
    """
    removed = 0
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            if fname.startswith("._"):
                ghost = os.path.join(dirpath, fname)
                try:
                    os.remove(ghost)
                    removed += 1
                except OSError:
                    pass
    return removed



    if smooth_ms <= 0:
        return arr
    sigma = (smooth_ms / 1000.0) / bin_size_s
    hw    = int(4 * sigma)
    x     = np.arange(-hw, hw + 1)
    k     = np.exp(-0.5 * (x / sigma) ** 2)
    k    /= k.sum()
    return np.convolve(arr, k, mode="same")


# ── Waveform feature functions ─────────────────────────────────────────

def wf_half_width(mean_wf: np.ndarray, t_axis: np.ndarray) -> float:
    """Width at half-amplitude of the trough (ms)."""
    trough_idx = np.argmin(mean_wf)
    half_amp   = mean_wf[trough_idx] / 2.0
    left  = np.where(mean_wf[:trough_idx] > half_amp)[0]
    right = np.where(mean_wf[trough_idx:] > half_amp)[0]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    return t_axis[trough_idx + right[0]] - t_axis[left[-1]]


def wf_peak_trough_ratio(mean_wf: np.ndarray) -> float:
    """
    Ratio of depolarisation peak (max before trough) to trough (min).
    Positive values near 1 = symmetric; < 1 = peak dominated; > 1 = trough dominated.
    """
    trough_idx = np.argmin(mean_wf)
    if trough_idx == 0:
        return float("nan")
    peak_val   = mean_wf[:trough_idx].max()
    trough_val = mean_wf[trough_idx]
    if trough_val == 0:
        return float("nan")
    return float(peak_val / abs(trough_val))


def wf_ahp_depth(mean_wf: np.ndarray, t_axis: np.ndarray,
                 ahp_window_ms: tuple = (1.0, 2.0)) -> float:
    """
    Mean amplitude in the AHP window (post-trough).
    Negative → hyperpolarisation; near 0 → fast-spiking.
    """
    mask = (t_axis >= ahp_window_ms[0]) & (t_axis <= ahp_window_ms[1])
    if not np.any(mask):
        return float("nan")
    return float(mean_wf[mask].mean())


def wf_repolarisation_slope(mean_wf: np.ndarray, t_axis: np.ndarray) -> float:
    """Mean slope (µV/ms) from trough to first positive zero-crossing after trough."""
    trough_idx = np.argmin(mean_wf)
    post = mean_wf[trough_idx:]
    t_post = t_axis[trough_idx:]
    cross = np.where(post >= 0)[0]
    if len(cross) == 0:
        cross_idx = len(post) - 1
    else:
        cross_idx = cross[0]
    if cross_idx == 0:
        return float("nan")
    return float((post[cross_idx] - post[0]) / (t_post[cross_idx] - t_post[0]))


def wf_trough_asymmetry(mean_wf: np.ndarray) -> float:
    """
    Ratio of area before trough to area after trough (absolute values).
    > 1 → slow rise, fast fall; < 1 → fast rise, slow fall.
    """
    trough_idx = np.argmin(mean_wf)
    area_before = np.sum(np.abs(mean_wf[:trough_idx]))
    area_after  = np.sum(np.abs(mean_wf[trough_idx:]))
    if area_after == 0:
        return float("nan")
    return float(area_before / area_after)


# ── Firing statistics ──────────────────────────────────────────────────

def firing_stats(spike_train_samples: np.ndarray,
                 fs: float,
                 duration_s: float,
                 fano_win_s: float) -> dict:
    """Return dict of scalar firing statistics."""
    if len(spike_train_samples) < 2:
        return {k: float("nan") for k in
                ["fr_mean_hz", "isi_cv", "isi_burst_pct",
                 "isi_mode_ms", "isi_viol_pct", "fano_factor"]}

    spike_times_s = spike_train_samples / fs
    isi_ms = np.diff(spike_times_s) * 1000.0

    fr_mean = len(spike_times_s) / duration_s

    cv = isi_ms.std() / isi_ms.mean() if isi_ms.mean() > 0 else float("nan")

    burst_pct = 100.0 * np.sum(isi_ms < 6.0)  / len(isi_ms)
    viol_pct  = 100.0 * np.sum(isi_ms < 3.0)  / len(isi_ms)

    # Mode of ISI (10 ms bins, range 0–200 ms)
    hist, edges = np.histogram(isi_ms, bins=np.arange(0, 201, 10))
    mode_ms = float(edges[np.argmax(hist)] + 5)   # bin centre

    # Fano factor: variance / mean of spike counts in windows
    win_samples = int(fano_win_s * fs)
    total_samples = int(duration_s * fs)
    n_wins = total_samples // win_samples
    counts = []
    for w in range(n_wins):
        t0 = w * fano_win_s
        t1 = t0 + fano_win_s
        counts.append(np.sum((spike_times_s >= t0) & (spike_times_s < t1)))
    counts = np.array(counts, dtype=float)
    fano = (counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")

    return {
        "fr_mean_hz"   : float(fr_mean),
        "isi_cv"       : float(cv),
        "isi_burst_pct": float(burst_pct),
        "isi_mode_ms"  : float(mode_ms),
        "isi_viol_pct" : float(viol_pct),
        "fano_factor"  : float(fano),
    }


# ══════════════════════════════════════════════════════════════════════
#  PROCESSING — ONE SESSION
# ══════════════════════════════════════════════════════════════════════

def process_session(session_name: str,
                    session_path: str,
                    ns6_files: list,
                    results_dir: str,
                    cache_dir: str) -> list:
    """
    Sort and extract features for every unit in all .ns6 files of a session.
    Returns a list of feature dicts (one per unit).
    """
    session_records = []

    for ns6_file in ns6_files:
        ns6_path    = os.path.join(session_path, ns6_file)
        base_name   = os.path.splitext(ns6_file)[0]
        sort_folder = os.path.join(cache_dir, session_name, base_name, "sorting")
        anal_folder = os.path.join(cache_dir, session_name, base_name, "analyzer")

        print(f"\n  ▸ {ns6_file}")

        # ── Load recording ───────────────────────────────────────────
        try:
            recording = se.BlackrockRecordingExtractor(ns6_path)
        except Exception as e:
            print(f"    ✗ Could not load: {e}")
            continue

        fs         = recording.get_sampling_frequency()
        duration_s = recording.get_duration()
        print(f"    fs={fs:.0f} Hz   dur={duration_s:.0f} s   "
              f"ch={recording.get_num_channels()}")

        recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
        recording.set_dummy_probe_from_locations(np.array([[0.0, 0.0]]))

        # ── Spike sorting (cached) ────────────────────────────────────
        sorting = None
        if os.path.isdir(sort_folder):
            # Purge any macOS ghost files before trying to load
            n_ghost = purge_macos_ghost_files(sort_folder)
            if n_ghost:
                print(f"    🧹 Removed {n_ghost} macOS ghost (._*) files from cache")
            print("    ↩ Loading cached sorting …")
            try:
                # tridesclous2 output is a SpikeGLX/NWB-style folder —
                # si.load_extractor() is the correct generic loader.
                # ss.NumpySorting.load() only works for NumpySorting objects.
                sorting = si.load_extractor(sort_folder)
            except Exception as e:
                print(f"    ⚠ Cache load failed ({e}) — re-running sorter …")
                sorting = None

        if sorting is None:
            print("    ⚙ Running spike sorter …")
            try:
                sorting = ss.run_sorter(
                    sorter_name="tridesclous2",
                    recording=recording,
                    folder=sort_folder,
                    remove_existing_folder=True,
                )
            except Exception as e:
                print(f"    ✗ Sorter failed: {e}")
                continue

        unit_ids = sorting.get_unit_ids()
        if len(unit_ids) == 0:
            print("    ⚠ No units found — skipping file.")
            continue
        print(f"    ✓ {len(unit_ids)} units found")

        # ── Sorting analyzer + waveforms ──────────────────────────────
        overwrite_anal = not os.path.isdir(anal_folder)
        try:
            analyzer = si.create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                format="binary_folder",
                folder=anal_folder,
                sparse=False,
                overwrite=overwrite_anal,
            )
            if overwrite_anal:
                analyzer.compute("random_spikes",
                                 method="uniform",
                                 max_spikes_per_unit=MAX_SPIKES_WF)
                analyzer.compute("waveforms",
                                 ms_before=WF_MS_BEFORE,
                                 ms_after=WF_MS_AFTER)
                analyzer.compute("templates",
                                 operators=["average", "median", "std"])
            waveforms_ext = analyzer.get_extension("waveforms")
        except Exception as e:
            print(f"    ✗ Analyzer failed: {e}")
            continue

        # Waveform time axis
        n_samples_wf = int((WF_MS_BEFORE + WF_MS_AFTER) / 1000.0 * fs)
        t_wf = np.linspace(-WF_MS_BEFORE, WF_MS_AFTER, n_samples_wf)

        # PCA of all mean waveforms in this file (for wf_pc* features)
        mean_wfs = []
        for uid in unit_ids:
            wf = waveforms_ext.get_waveforms_one_unit(uid)[:, :, 0]
            mean_wfs.append(wf.mean(axis=0)[:n_samples_wf])

        mean_wfs_arr = np.vstack(mean_wfs)
        wf_scaler    = StandardScaler()
        mean_wfs_sc  = wf_scaler.fit_transform(mean_wfs_arr)
        n_pc = min(5, len(unit_ids))
        pca_wf       = PCA(n_components=n_pc, random_state=RANDOM_STATE)
        wf_pcs       = pca_wf.fit_transform(mean_wfs_sc)   # (n_units, n_pc)

        depth_hint = extract_depth_from_filename(ns6_file)

        # ── Per-unit features ─────────────────────────────────────────
        wf_plot_dir = os.path.join(results_dir, "per_unit_waveforms")
        os.makedirs(wf_plot_dir, exist_ok=True)

        for u_idx, unit_id in enumerate(unit_ids):
            spike_train = sorting.get_unit_spike_train(
                unit_id=unit_id, segment_index=0
            )
            n_spikes = len(spike_train)

            if n_spikes < MIN_SPIKES:
                print(f"    · Unit {unit_id}: {n_spikes} spikes < {MIN_SPIKES} → skip")
                continue

            # Waveforms
            wf_all = waveforms_ext.get_waveforms_one_unit(unit_id)[:, :, 0]
            t_plot = t_wf[:wf_all.shape[1]]
            mean_wf = wf_all.mean(axis=0)
            std_wf  = wf_all.std(axis=0)

            # Waveform features
            feat = {
                "session"          : session_name,
                "file"             : base_name,
                "unit_id"          : int(unit_id),
                "n_spikes"         : int(n_spikes),
                "depth_um"         : float(depth_hint),
                "wf_width_ms"      : wf_half_width(mean_wf, t_plot),
                "wf_peak_trough"   : wf_peak_trough_ratio(mean_wf),
                "wf_ahp_depth"     : wf_ahp_depth(mean_wf, t_plot),
                "wf_repol_slope"   : wf_repolarisation_slope(mean_wf, t_plot),
                "wf_trough_asym"   : wf_trough_asymmetry(mean_wf),
            }

            # PCA components of this unit's mean waveform
            for pc_i in range(n_pc):
                feat[f"wf_pc{pc_i+1}"] = float(wf_pcs[u_idx, pc_i])
            # Pad missing PCs if fewer units than 5
            for pc_i in range(n_pc, 5):
                feat[f"wf_pc{pc_i+1}"] = float("nan")

            # Firing statistics
            feat.update(firing_stats(spike_train, fs, duration_s, FANO_WIN_S))

            session_records.append(feat)

            # ── Per-unit waveform + ISI figure ────────────────────────
            _plot_unit(
                unit_id=unit_id,
                base_name=base_name,
                session_name=session_name,
                wf_all=wf_all,
                mean_wf=mean_wf,
                std_wf=std_wf,
                t_plot=t_plot,
                spike_train=spike_train,
                fs=fs,
                feat=feat,
                out_dir=wf_plot_dir,
                max_wf=MAX_WF_PLOT,
            )

    return session_records


# ══════════════════════════════════════════════════════════════════════
#  PER-UNIT FIGURE
# ══════════════════════════════════════════════════════════════════════

def _plot_unit(unit_id, base_name, session_name,
               wf_all, mean_wf, std_wf, t_plot,
               spike_train, fs, feat, out_dir, max_wf):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    fig.suptitle(
        f"{session_name}  |  {base_name}  |  Unit {unit_id}"
        f"  |  {feat['n_spikes']} spikes",
        fontsize=9, fontweight="bold"
    )

    # Waveform
    ax = axes[0]
    for spike in wf_all[:max_wf]:
        ax.plot(t_plot, spike, color="grey", alpha=0.15, linewidth=0.4)
    ax.fill_between(t_plot, mean_wf - std_wf, mean_wf + std_wf,
                    color="steelblue", alpha=0.25, linewidth=0)
    ax.plot(t_plot, mean_wf, color="steelblue", linewidth=2.0)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.0, label="spike peak")

    w = feat["wf_width_ms"]
    pt = feat["wf_peak_trough"]
    info = (f"width = {w:.2f} ms\n" if not np.isnan(w) else "") + \
           (f"P/T = {pt:.2f}" if not np.isnan(pt) else "")
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="steelblue")
    ax.set_xlabel("Time relative to spike peak (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Waveform (grey=individual, blue=mean±std)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)

    # ISI
    ax2 = axes[1]
    isi_ms = np.diff(spike_train) / fs * 1000
    viol_pct = feat["isi_viol_pct"]
    ax2.hist(isi_ms, bins=100, range=(0, 100),
             color="steelblue", edgecolor="white", linewidth=0.3)
    ax2.axvline(3, color="red", linestyle="--", linewidth=1.5,
                label=f"3 ms refractory\n{viol_pct:.1f}% violations")
    ax2.set_xlabel("ISI (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"ISI  |  mean FR = {feat['fr_mean_hz']:.1f} Hz"
                  f"  |  CV = {feat['isi_cv']:.2f}")
    ax2.legend(fontsize=7, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, f"{session_name}_{base_name}_unit{unit_id}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  CLUSTERING
# ══════════════════════════════════════════════════════════════════════

# Features used for clustering (must be numeric, no NaN after imputation)
CLUSTER_FEATURES = [
    "wf_width_ms",
    "wf_peak_trough",
    "wf_ahp_depth",
    "wf_repol_slope",
    "wf_trough_asym",
    "wf_pc1", "wf_pc2", "wf_pc3",
    "fr_mean_hz",
    "isi_cv",
    "isi_burst_pct",
    "fano_factor",
]


def prepare_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Returns scaled feature matrix X_scaled, list of valid row indices,
    and the scaler object.
    """
    sub = df[CLUSTER_FEATURES].copy()
    # Impute NaN with column median (robust to outliers)
    for col in sub.columns:
        med = sub[col].median()
        sub[col] = sub[col].fillna(med)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(sub.values)
    return X_scaled, scaler


def fit_gmm_sweep(X: np.ndarray,
                  max_k: int,
                  random_state: int) -> tuple:
    """
    Fit GMMs for k = 1..max_k.
    Returns arrays of k values, BIC values, silhouette scores, and all fitted models.
    """
    ks, bics, sils, models = [], [], [], []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            max_iter=300,
            n_init=10,
            random_state=random_state,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        bic    = gmm.bic(X)
        sil    = silhouette_score(X, labels) if k > 1 else float("nan")

        ks.append(k);  bics.append(bic);  sils.append(sil);  models.append(gmm)
        print(f"    k={k:2d}  BIC={bic:10.1f}  sil={sil:.3f}")

    return np.array(ks), np.array(bics), np.array(sils), models


# ══════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════

CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def plot_bic(ks, bics, best_k, out_path):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.plot(ks, bics, "o-", color="steelblue", linewidth=2)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
               label=f"best k = {best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title("GMM model selection — BIC curve")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_pca_overview(X_scaled, labels, df, n_units, out_path):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X2   = pca2.fit_transform(X_scaled)
    ev   = pca2.explained_variance_ratio_

    unique_clusters = np.unique(labels)
    sessions        = df["session"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    # Left — colour by cluster
    ax = axes[0]
    for c in unique_clusters:
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=CLUSTER_COLORS[c % 10],
                   alpha=0.65, s=25, label=f"Cluster {c+1}",
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_title("PCA — coloured by cluster")
    ax.legend(markerscale=1.5, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Right — colour by session
    ax2   = axes[1]
    sess  = sorted(df["session"].unique())
    cmap2 = plt.cm.Set2(np.linspace(0, 1, len(sess)))
    for si_idx, s in enumerate(sess):
        mask = sessions == s
        ax2.scatter(X2[mask, 0], X2[mask, 1],
                    color=cmap2[si_idx],
                    alpha=0.65, s=25, label=s,
                    edgecolors="white", linewidths=0.3)
    ax2.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax2.set_title("PCA — coloured by session")
    ax2.legend(markerscale=1.5, fontsize=7, frameon=False,
               bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"LGN Cell Classification  |  {n_units} units",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_umap_overview(X_scaled, labels, df, n_units, out_path):
    try:
        import umap
    except ImportError:
        print("  ⚠ umap-learn not installed — skipping UMAP plot. "
              "Install with: pip install umap-learn")
        return

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=15, min_dist=0.1)
    X2 = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=CLUSTER_COLORS[c % 10],
                   alpha=0.70, s=28, label=f"Cluster {c+1}",
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP — {n_units} units")
    ax.legend(markerscale=1.5, fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_feature_distributions(df: pd.DataFrame, out_path: str):
    """
    Box-plots of key features split by cluster, so you can see what
    drives the separation.
    """
    plot_feats = [
        ("wf_width_ms",    "Spike half-width (ms)"),
        ("wf_peak_trough", "Peak / Trough ratio"),
        ("wf_ahp_depth",   "AHP depth (µV)"),
        ("fr_mean_hz",     "Mean firing rate (Hz)"),
        ("isi_cv",         "ISI coefficient of variation"),
        ("isi_burst_pct",  "Burst ISIs < 6 ms (%)"),
    ]
    clusters = sorted(df["cluster"].unique())
    n_feat   = len(plot_feats)
    ncols    = 3
    nrows    = int(np.ceil(n_feat / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             facecolor="white")
    axes = axes.flatten()

    for i, (feat_col, feat_label) in enumerate(plot_feats):
        ax = axes[i]
        data_by_cluster = [
            df.loc[df["cluster"] == c, feat_col].dropna().values
            for c in clusters
        ]
        bp = ax.boxplot(data_by_cluster,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], clusters):
            patch.set_facecolor(CLUSTER_COLORS[c % 10])
            patch.set_alpha(0.7)
        ax.set_xticklabels([f"C{c+1}" for c in clusters])
        ax.set_title(feat_label, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused panels
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions by cluster", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_waveform_gallery(df: pd.DataFrame, out_path: str):
    """
    For each cluster, overlay the mean waveforms (one colour per cluster).
    Useful sanity check that clusters capture real waveform differences.
    Note: only works if we store the mean waveform; here we use PCs as proxy
    and show a scatter of wf_pc1 vs wf_width_ms instead.
    """
    clusters = sorted(df["cluster"].unique())
    fig, ax  = plt.subplots(figsize=(7, 5), facecolor="white")

    for c in clusters:
        sub = df[df["cluster"] == c]
        ax.scatter(sub["wf_width_ms"], sub["fr_mean_hz"],
                   color=CLUSTER_COLORS[c % 10],
                   alpha=0.65, s=35, label=f"Cluster {c+1}",
                   edgecolors="white", linewidths=0.3)

    ax.set_xlabel("Spike half-width (ms)")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Width vs Firing Rate — primary classification axes")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)

    # Pre-emptive ghost file sweep on the cache root (local, but just in case)
    n = purge_macos_ghost_files(CACHE_DIR)
    if n:
        print(f"  🧹 Removed {n} macOS ghost files from cache root")

    # ── 1. Discover session folders ────────────────────────────────────
    print("\n╔══════════════════════════════════════════╗")
    print("║  LGN Cell-Type Classification Pipeline  ║")
    print("╚══════════════════════════════════════════╝\n")

    sessions = find_session_folders(ELECTROPHYSIOLOGY_ROOT)
    print(f"Found {len(sessions)} session folders:\n")
    for name, _, files in sessions:
        print(f"  {name}/  ({len(files)} .ns6 files)")

    if not sessions:
        raise RuntimeError(
            f"No session folders with .ns6 files found under:\n  {ELECTROPHYSIOLOGY_ROOT}"
        )

    # ── 2. Extract features for every session ─────────────────────────
    all_records = []
    for session_name, session_path, ns6_files in sessions:
        print(f"\n{'─'*55}")
        print(f"  SESSION: {session_name}  ({len(ns6_files)} files)")
        print(f"{'─'*55}")
        records = process_session(
            session_name=session_name,
            session_path=session_path,
            ns6_files=ns6_files,
            results_dir=RESULTS_DIR,
            cache_dir=CACHE_DIR,
        )
        all_records.extend(records)
        print(f"  → {len(records)} units extracted")

    if not all_records:
        raise RuntimeError("No units extracted. Check sorter output and MIN_SPIKES setting.")

    df = pd.DataFrame(all_records)
    n_units = len(df)
    print(f"\n{'═'*55}")
    print(f"  Total units: {n_units} across {df['session'].nunique()} sessions")
    print(f"{'═'*55}\n")

    # Save raw features
    feat_csv = os.path.join(RESULTS_DIR, "features.csv")
    df.to_csv(feat_csv, index=False)
    print(f"  Features saved → {feat_csv}")

    # ── 3. Prepare feature matrix ──────────────────────────────────────
    X_scaled, scaler = prepare_feature_matrix(df)

    # ── 4. GMM sweep — pick best k ────────────────────────────────────
    print("\n  Fitting GMMs …")
    ks, bics, sils, models = fit_gmm_sweep(X_scaled, MAX_K, RANDOM_STATE)

    # Best k = elbow of BIC (first k where improvement < 5% of total range)
    bic_range = bics.max() - bics.min()
    best_k_idx = 0
    for i in range(1, len(bics)):
        if bics[i - 1] - bics[i] < 0.05 * bic_range:
            best_k_idx = i - 1
            break
        best_k_idx = i
    best_k   = int(ks[best_k_idx])
    best_gmm = models[best_k_idx]

    print(f"\n  ✓ Best k = {best_k}  (BIC elbow)")

    labels = best_gmm.predict(X_scaled)
    probs  = best_gmm.predict_proba(X_scaled)

    df["cluster"]          = labels
    df["cluster_prob_max"] = probs.max(axis=1)   # assignment confidence

    # ── 5. Cluster summary ─────────────────────────────────────────────
    summary_cols = CLUSTER_FEATURES + ["n_spikes", "depth_um"]
    summary = df.groupby("cluster")[summary_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(RESULTS_DIR, "cluster_summary.csv"))

    print("\n  Cluster sizes:")
    for c in sorted(df["cluster"].unique()):
        n = (df["cluster"] == c).sum()
        w = df.loc[df["cluster"] == c, "wf_width_ms"].median()
        fr = df.loc[df["cluster"] == c, "fr_mean_hz"].median()
        print(f"    Cluster {c+1}: {n:3d} units  "
              f"median width={w:.2f} ms  median FR={fr:.1f} Hz")

    # Save updated features with cluster labels
    df.to_csv(feat_csv, index=False)

    # ── 6. Plots ───────────────────────────────────────────────────────
    print("\n  Generating plots …")

    plot_bic(ks, bics, best_k,
             os.path.join(RESULTS_DIR, "bic_curve.png"))

    plot_pca_overview(X_scaled, labels, df, n_units,
                      os.path.join(RESULTS_DIR, "pca_overview.png"))

    plot_umap_overview(X_scaled, labels, df, n_units,
                       os.path.join(RESULTS_DIR, "umap_overview.png"))

    plot_feature_distributions(df,
                               os.path.join(RESULTS_DIR, "feature_distributions.png"))

    plot_waveform_gallery(df,
                          os.path.join(RESULTS_DIR, "width_vs_fr.png"))

    print(f"\n  ✓ All done. Results in:\n    {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()