
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io

from sklearn.svm            import OneClassSVM, SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.metrics        import (confusion_matrix, classification_report,
                                    roc_auc_score, roc_curve,
                                    ConfusionMatrixDisplay)
from sklearn.utils          import resample

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
CAMELOT_FEATURES_CSV = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/"
    "Monkey/Camelot/ml_results/features.csv"
)
REFERENCE_DIR = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/"
    "Monkey/Rapid adaptation of primate LGN neurons to drifting grating stimulation/"
    "10.12751_g-node.kvut7v/data"
)

ML_RESULTS_DIR    = os.path.dirname(CAMELOT_FEATURES_CSV)
MANUAL_LABELS_CSV = os.path.join(ML_RESULTS_DIR, "manual_labels.csv")
RESULTS_DIR       = os.path.join(ML_RESULTS_DIR, "lgn_classification")

# ══════════════════════════════════════════════════════════════════════
#  FEATURES  — waveform shape only (task-independent)
# ══════════════════════════════════════════════════════════════════════
# Features used for CROSS-DATASET comparison (Camelot vs Daumail 2023).
# Only 3 features are used here
#   wf_ahp_depth   :stored as raw ADC counts in Daumail, µV in Camelot
#                    : normalisation collapses Camelot median to 0 (too many NaN)
#   wf_repol_slope : same problem
# The 3 features below are either time-based (ms) or pure ratios,
# so they are truly amplitude-invariant across datasets.
WF_FEATURES_CROSS = [
    "wf_width_ms",       # spike half-width (ms) — time-based, scale-free
    "wf_peak_trough",    # peak/trough ratio — dimensionless
    "wf_trough_asym",    # area ratio — dimensionless
]

# Features used for WITHIN-CAMELOT supervised SVM (Strategy B).
# All 5 are valid here because both labelled and unlabelled cells come
# from the same recording system (same units, same amplifier gain).
WF_FEATURES_SUPER = [
    "wf_width_ms",
    "wf_peak_trough",
    "wf_ahp_depth",
    "wf_repol_slope",
    "wf_trough_asym",
]

# Alias used in print statements and CSV output
WF_FEATURES = WF_FEATURES_SUPER   # full set shown in headers

FS_NEURAL = 30_000.0    # Blackrock NSP — confirmed from Daumail onsets data

# ══════════════════════════════════════════════════════════════════════
#  WAVEFORM FEATURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def wf_half_width(w, t):
    ti = np.argmin(w)
    ha = w[ti] / 2.0
    L  = np.where(w[:ti] > ha)[0]
    R  = np.where(w[ti:] > ha)[0]
    return float(t[ti + R[0]] - t[L[-1]]) if len(L) and len(R) else float("nan")

def wf_peak_trough(w):
    ti = np.argmin(w)
    if ti == 0 or w[ti] == 0:
        return float("nan")
    return float(w[:ti].max() / abs(w[ti]))

def wf_ahp(w, t, win=(1.0, 2.0)):
    m = (t >= win[0]) & (t <= win[1])
    return float(w[m].mean()) if m.any() else float("nan")

def wf_slope(w, t):
    ti   = np.argmin(w)
    post = w[ti:];  tp = t[ti:]
    ci   = np.where(post >= 0)[0]
    ci   = ci[0] if len(ci) else len(post) - 1
    if ci == 0:
        return float("nan")
    return float((post[ci] - post[0]) / (tp[ci] - tp[0]))

def wf_asym(w):
    ti = np.argmin(w)
    a  = np.sum(np.abs(w[ti:]))
    return float(np.sum(np.abs(w[:ti])) / a) if a > 0 else float("nan")

def normalise_waveform(w: np.ndarray) -> np.ndarray:
    """
    Z-score normalise so only SHAPE is compared, not amplitude.
    Critical for cross-dataset use: Daumail stores raw int16 ADC counts,
    Camelot features are in µV — without this, wf_ahp_depth and
    wf_repol_slope are on completely different scales.
    After normalisation: mean=0, std=1 for every waveform.
    """
    std = w.std()
    return (w - w.mean()) / std if std > 1e-9 else w - w.mean()


def wf_features(mean_wf, t_wf):
    w = normalise_waveform(mean_wf)   # shape-only, amplitude-invariant
    return {
        "wf_width_ms"   : wf_half_width(w, t_wf),
        "wf_peak_trough": wf_peak_trough(w),
        "wf_ahp_depth"  : wf_ahp(w, t_wf),
        "wf_repol_slope": wf_slope(w, t_wf),
        "wf_trough_asym": wf_asym(w),
    }

# ══════════════════════════════════════════════════════════════════════
#  DAUMAIL 2023 MAT LOADER
# ══════════════════════════════════════════════════════════════════════

def load_reference_mat(path):
    """Return one feature dict from a Daumail 2023 .mat file, or {}."""
    try:
        raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return {}

    cd = raw.get("channel_data")
    if not hasattr(cd, "_fieldnames"):
        return {}
    wf_s = getattr(cd, "wf", None)
    if not hasattr(wf_s, "_fieldnames"):
        return {}

    wf_raw = getattr(wf_s, "waveForms", None)
    if wf_raw is None:
        return {}

    wf_arr = np.array(wf_raw, dtype=float)
    if wf_arr.ndim != 3:
        return {}

    n_spikes, n_ch, n_samp = wf_arr.shape
    mean_ch   = wf_arr.mean(axis=0)                     # (n_ch, n_samp)
    peak_ch   = int(np.argmax(mean_ch.max(1) - mean_ch.min(1)))
    mean_wf   = mean_ch[peak_ch] - mean_ch[peak_ch].mean()

    ti        = int(np.argmin(mean_wf))
    t_wf      = np.linspace(-ti/FS_NEURAL*1000,
                             (n_samp-ti-1)/FS_NEURAL*1000, n_samp)

    feat = {"source": os.path.basename(path), "n_spikes": n_spikes}
    feat.update(wf_features(mean_wf, t_wf))
    return feat if not np.isnan(feat["wf_width_ms"]) else {}

# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def impute_scale(df, feats, scaler=None):
    sub = df[feats].copy()
    for c in sub.columns:
        med = sub[c].median()
        sub[c] = sub[c].fillna(0.0 if np.isnan(med) else med)
    X = sub.values.astype(float)
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


def lofo_cv(df, feats, label_col="is_lgn"):
    """
    Leave-One-File-Out Cross-Validation.
    Holds out ALL units from one file at a time — avoids leakage
    from correlated units recorded on the same electrode.
    Returns (y_true, y_pred, y_prob) as plain numpy arrays.
    """
    # Always work on a clean 0-based integer index
    df = df.reset_index(drop=True)
    files = df["file"].unique()

    # Accumulate into plain lists — no index gymnastics
    all_true, all_pred, all_prob = [], [], []
    skipped = []

    for test_file in files:
        test_mask  = df["file"] == test_file
        train_df   = df[~test_mask]
        test_df    = df[test_mask]

        n_classes_train = train_df[label_col].nunique()
        if n_classes_train < 2:
            skipped.append(
                f"{test_file} (training set has only "
                f"{n_classes_train} class after holdout)"
            )
            continue

        X_tr, sc = impute_scale(train_df, feats)
        X_te, _  = impute_scale(test_df,  feats, scaler=sc)
        y_tr     = train_df[label_col].values.astype(int)

        clf = SVC(kernel="rbf", probability=True,
                  class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)

        all_true.extend(test_df[label_col].values.tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())

    if skipped:
        print(f"    ⚠ {len(skipped)} files skipped in LOFO-CV "
              f"(training set missing one class):")
        for s in skipped[:5]:
            print(f"      {s}")
        if len(skipped) > 5:
            print(f"      … and {len(skipped)-5} more")

    if not all_true:
        return np.array([]), np.array([]), np.array([])

    return (np.array(all_true,  dtype=int),
            np.array(all_pred,  dtype=int),
            np.array(all_prob,  dtype=float))

# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════

C_REF    = np.array([0.13, 0.47, 0.71])
C_LGN    = np.array([0.12, 0.63, 0.36])
C_NONLGN = np.array([0.80, 0.15, 0.15])


def plot_confusion_and_roc(y_true, y_pred, y_prob, n_lgn, n_nonlgn, out_path):
    """
    Two-panel figure:
      Left  — confusion matrix  (from LOFO-CV)
      Right — ROC curve         (from LOFO-CV probabilities)
    """
    cm  = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # ── Confusion matrix ────────────────────────────────────────────
    ax = axes[0]
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-LGN", "LGN"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc  = (tp + tn) / cm.sum() if cm.sum() > 0 else 0

    ax.set_title(
        f"Confusion matrix — Leave-One-File-Out CV\n"
        f"Sensitivity = {sens:.2f}  |  Specificity = {spec:.2f}  |  "
        f"Accuracy = {acc:.2f}",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)

    # Add per-cell percentage annotations
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / total
            ax.text(j, i + 0.3, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=8, color="grey")

    # ── ROC curve ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color=C_LGN, linewidth=2.5,
             label=f"ROC curve  (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="grey", linestyle="--",
             linewidth=1, label="Random classifier")
    ax2.fill_between(fpr, tpr, alpha=0.12, color=C_LGN)
    ax2.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=9)
    ax2.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=9)
    ax2.set_title(
        f"ROC — LOFO-CV\n"
        f"Labelled: {n_lgn} LGN units  |  {n_nonlgn} non-LGN units",
        fontsize=9, fontweight="bold"
    )
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)

    # AUC interpretation guide
    guide = ("AUC interpretation:\n"
             "  0.5  = random\n"
             "  0.7+ = acceptable\n"
             "  0.8+ = good\n"
             "  0.9+ = excellent")
    ax2.text(0.60, 0.22, guide, transform=ax2.transAxes,
             fontsize=7.5, color="grey",
             bbox=dict(facecolor="white", edgecolor="lightgrey",
                       boxstyle="round,pad=0.4"))

    plt.suptitle("Strategy B — Supervised classifier performance (LOFO-CV)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return auc, sens, spec, acc


def plot_wf_pca(X_ref, X_lab_lgn, X_lab_non, X_unlab, unlab_proba,
                pca, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    ev = pca.explained_variance_ratio_

    titles = [
        "Reference LGN + Camelot manual labels",
        "All Camelot units (colour = P(LGN) for unlabelled)",
    ]

    for ax_idx, ax in enumerate(axes):
        # Reference LGN
        p = pca.transform(X_ref)
        ax.scatter(p[:,0], p[:,1], color=C_REF, s=18, alpha=0.5,
                   marker="o", label="Daumail 2023 LGN ref",
                   edgecolors="none", zorder=2)

        if ax_idx == 0:
            # Manual labels only
            if len(X_lab_lgn):
                p = pca.transform(X_lab_lgn)
                ax.scatter(p[:,0], p[:,1], color=C_LGN, s=70, alpha=0.85,
                           marker="^", edgecolors="white", linewidths=0.5,
                           label=f"Camelot manual LGN (n={len(X_lab_lgn)})",
                           zorder=5)
            if len(X_lab_non):
                p = pca.transform(X_lab_non)
                ax.scatter(p[:,0], p[:,1], color=C_NONLGN, s=70, alpha=0.85,
                           marker="v", edgecolors="white", linewidths=0.5,
                           label=f"Camelot manual non-LGN (n={len(X_lab_non)})",
                           zorder=5)
        else:
            # All Camelot unlabelled, coloured by P(LGN)
            if len(X_unlab):
                p  = pca.transform(X_unlab)
                sc = ax.scatter(p[:,0], p[:,1], c=unlab_proba,
                                cmap="RdYlGn", vmin=0, vmax=1,
                                s=22, alpha=0.65, marker="o",
                                edgecolors="none",
                                label="Unlabelled — colour=P(LGN)",
                                zorder=3)
                plt.colorbar(sc, ax=ax, label="P(LGN)", shrink=0.85)

        ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)", fontsize=9)
        ax.set_title(titles[ax_idx], fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_feature_boxes(ref_df, cam_final, out_path):
    lgn_c    = cam_final[cam_final["final_lgn"] == 1]
    nonlgn_c = cam_final[cam_final["final_lgn"] == 0]

    feat_labels = [
        ("wf_width_ms",    "Spike half-width (ms)\nnarrow=M/interneuron, broad=P"),
        ("wf_peak_trough", "Peak / Trough ratio"),
        ("wf_ahp_depth",   "AHP depth (µV)"),
        ("wf_repol_slope", "Repol. slope (µV/ms)"),
        ("wf_trough_asym", "Trough asymmetry"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), facecolor="white")
    for ax, (col, lbl) in zip(axes, feat_labels):
        groups = [ref_df[col].dropna().values,
                  lgn_c[col].dropna().values,
                  nonlgn_c[col].dropna().values]
        valid_g = [g for g in groups if len(g) > 0]
        colors  = [C_REF, C_LGN, C_NONLGN]
        tlbls   = ["Ref\nLGN", "Camelot\nLGN", "Camelot\nnon-LGN"]
        valid_c = [c for g, c in zip(groups, colors)   if len(g) > 0]
        valid_l = [l for g, l in zip(groups, tlbls)    if len(g) > 0]

        bp = ax.boxplot(valid_g, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], valid_c):
            patch.set_facecolor(c); patch.set_alpha(0.70)
        ax.set_xticklabels(valid_l, fontsize=8)
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Waveform features by classification group",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_session_summary(cam_final, out_path):
    """Bar chart: LGN / non-LGN counts per session."""
    summary = (cam_final.groupby(["session", "final_lgn"])
               .size().unstack(fill_value=0))
    if 1 not in summary.columns:
        summary[1] = 0
    if 0 not in summary.columns:
        summary[0] = 0

    fig, ax = plt.subplots(figsize=(max(8, len(summary)*0.8), 5),
                           facecolor="white")
    x = np.arange(len(summary))
    w = 0.38
    ax.bar(x - w/2, summary[1].values, width=w,
           color=C_LGN,    alpha=0.80, label="LGN")
    ax.bar(x + w/2, summary[0].values, width=w,
           color=C_NONLGN, alpha=0.80, label="non-LGN")

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Number of units")
    ax.set_title("LGN vs non-LGN units per session", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n╔═══════════════════════════════════════════════════╗")
    print("║   LGN vs non-LGN Classifier                    ║")
    print("║   Features: waveform shape only (task-agnostic)  ║")
    print("╚═══════════════════════════════════════════════════╝\n")

    # ── Load Camelot features ──────────────────────────────────────────
    if not os.path.isfile(CAMELOT_FEATURES_CSV):
        print(f"✗ Not found: {CAMELOT_FEATURES_CSV}"); sys.exit(1)

    cam = pd.read_csv(CAMELOT_FEATURES_CSV)
    print(f"Camelot dataset: {len(cam)} units  |  "
          f"{cam['session'].nunique()} sessions  |  "
          f"{cam['file'].nunique()} files\n")

    # Check which WF features are present
    missing = [f for f in WF_FEATURES_SUPER if f not in cam.columns]
    if missing:
        print(f"\u2717 Missing features in features.csv: {missing}")
        print("  Re-run lgn_cell_classifier.py to regenerate features.")
        sys.exit(1)

    # ── Per-unit waveform normalisation ───────────────────────────────
    # The features in features.csv were extracted from µV waveforms.
    # The Daumail reference features are extracted from raw ADC counts.
    # Re-normalise the 5 shape features to z-scores (per-unit) so the
    # two datasets live on the same scale before OC-SVM comparison.
    #
    # We do this by constructing a 5-feature vector per unit, normalising
    # it to unit norm, then replacing the columns.  This preserves the
    # RATIOS between features (shape) while removing absolute scale.
    #
    # Note: wf_width_ms is already in ms (time-based, scale-invariant).
    # wf_trough_asym is a ratio (scale-invariant).
    # wf_peak_trough is a ratio (scale-invariant).
    # Only wf_ahp_depth and wf_repol_slope are amplitude-dependent.
    # We are normalise those two specifically.
    for amp_feat in ["wf_ahp_depth", "wf_repol_slope"]:
        if amp_feat in cam.columns:
            col  = cam[amp_feat].dropna()
            med  = col.median()
            mad  = (col - med).abs().median()
            if mad > 1e-9:
                cam[amp_feat] = (cam[amp_feat] - med) / mad
    print("  Amplitude-dependent features (wf_ahp_depth, wf_repol_slope) "
          "normalised to median/MAD scale.\n")

    # ════════════════════════════════════════════════════════════════
    #   — One-Class SVM on Daumail 2023 reference
    # ════════════════════════════════════════════════════════════════
    print("═"*55)
    print("STRATEGY A — Cross-dataset waveform matching (OC-SVM)")
    print("  Features:", WF_FEATURES)
    print("═"*55 + "\n")

    mat_files = []
    for root, dirs, files in os.walk(REFERENCE_DIR):
        dirs[:] = [d for d in dirs
                   if not d.startswith("._") and d != "__MACOSX"]
        for f in files:
            if f.endswith(".mat") and not f.startswith("._"):
                mat_files.append(os.path.join(root, f))

    print(f"  Found {len(mat_files)} reference mat files …")

    ref_records = [r for r in
                   (load_reference_mat(p) for p in sorted(mat_files)) if r]
    ref_df = pd.DataFrame(ref_records) if ref_records else pd.DataFrame()

    strategy_a_scores = None
    strategy_a_pred   = None

    if ref_df.empty:
        print("  ⚠ No reference features extracted. "
              "Check REFERENCE_DIR. Strategy A skipped.")
    else:
        print(f"  ✓ {len(ref_df)} reference LGN units extracted\n")

        # Feature summary
        print(f"  {'Feature':<20}  {'Ref LGN median':>15}  "
              f"{'Camelot median':>15}")
        print(f"  {'─'*20}  {'─'*15}  {'─'*15}")
        for f in WF_FEATURES_CROSS:
            rv = ref_df[f].median() if f in ref_df else float("nan")
            cv = cam[f].median()   if f in cam   else float("nan")
            print(f"  {f:<20}  {rv:>15.3f}  {cv:>15.3f}")
        print()

        X_ref, sc_a   = impute_scale(ref_df, WF_FEATURES_CROSS)
        X_cam_a, sc_a = impute_scale(cam,    WF_FEATURES_CROSS, scaler=sc_a)

        oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        oc.fit(X_ref)

        n_ref_in = int((oc.predict(X_ref) == 1).sum())
        print(f"  Boundary check: {n_ref_in}/{len(ref_df)} reference units "
              f"inside boundary  (expected ≥ {int(0.95*len(ref_df))})")

        strategy_a_scores = oc.decision_function(X_cam_a)
        strategy_a_pred   = (oc.predict(X_cam_a) == 1).astype(int)

        n_a = int(strategy_a_pred.sum())
        print(f"  Result: {n_a} LGN  |  {len(strategy_a_pred)-n_a} non-LGN  "
              f"({100*n_a/len(strategy_a_pred):.1f}% LGN)\n")

    # ════════════════════════════════════════════════════════════════
    #  Supervised SVM from manual labels
    # ════════════════════════════════════════════════════════════════
    print("═"*55)
    print("STRATEGY B — Supervised SVM (manual labels)")
    print("  Features:", WF_FEATURES)
    print("  CV method: Leave-One-File-Out")
    print("═"*55 + "\n")

    strategy_b_scores = None
    strategy_b_pred   = None
    lofo_auc          = float("nan")
    labelled_df       = pd.DataFrame()
    unlabelled_df     = pd.DataFrame()
    unlab_proba_arr   = np.array([])

    if not os.path.isfile(MANUAL_LABELS_CSV):
        print(f"  ⚠ {MANUAL_LABELS_CSV} not found.")
        print("  Create manual_labels.csv to enable Strategy B and get a")
        print("  confusion matrix.\n")
        print("  Format example:")
        print("    file,is_lgn")
        print("    cam_20260330_25350_righteyeFF_006,1")
        print("    cam_20260408_28200_MOTIONTASK_001,0\n")
    else:
        # ── Manually parse the CSV/TSV — handles all quoting variants ──
        # Some apps (Excel, Numbers) wrap entire rows in double quotes,
        # e.g.  "file\tis_lgn"  instead of  file\tis_lgn
        # pandas cannot handle this; we parse line-by-line instead.

        def _parse_labels_file(path):
            """
            Robust parser for manual_labels.csv.
            Handles three malformed formats that spreadsheet apps produce:

              Format 1 — correct:
                file,is_lgn
                cam_..._006,1

              Format 2 — whole row quoted with tab inside:
                "cam_..._024<TAB>1"

              Format 3 — label glued to filename with no separator:
                cam_..._0421   (last char is the 0/1 label)
            """
            with open(path, "rb") as _fh:
                raw_bytes = _fh.read()
            text  = raw_bytes.decode("utf-8-sig")  # strip BOM
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if not lines:
                return pd.DataFrame()

            records = []
            for line in lines[1:]:   # skip header row

                # Format 2: entire row quoted, separator is tab inside quotes
                if line.startswith('"') and line.endswith('"'):
                    inner = line[1:-1]
                    if "\t" in inner:
                        parts = inner.split("\t")
                        if len(parts) == 2:
                            records.append((parts[0].strip(), parts[1].strip()))
                            continue

                # Format 1: normal comma or tab separated
                if "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        records.append((parts[0].strip(), parts[1].strip()))
                        continue
                if "," in line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        records.append((parts[0].strip(), parts[1].strip()))
                        continue

                # Format 3: label (0 or 1) glued to end of filename
                if len(line) > 1 and line[-1] in ("0", "1"):
                    records.append((line[:-1].strip(), line[-1]))
                    continue

                print(f"    ⚠ Could not parse row: {repr(line)}")

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records, columns=["file", "is_lgn"])
            df["file"]   = df["file"].str.strip()
            df["is_lgn"] = pd.to_numeric(df["is_lgn"], errors="coerce")
            return df

        _ldf = _parse_labels_file(MANUAL_LABELS_CSV)
        print(f"  Rows parsed : {len(_ldf)}")

        if len(_ldf) == 0:
            print("  ✗ Could not parse any rows from manual_labels.csv.")
            sys.exit(1)

        # Column names are always 'file' and 'is_lgn' from the parser above
        _ldf["_file_clean"] = (_ldf["file"]
                                .str.replace(r"\.ns6$", "", regex=True)
                                .str.replace(r"\.mat$", "", regex=True)
                                .str.strip())

        labels_map = _ldf.set_index("_file_clean")["is_lgn"].dropna().to_dict()
        n_lgn_csv    = int(sum(1 for v in labels_map.values() if v == 1))
        n_nonlgn_csv = int(sum(1 for v in labels_map.values() if v == 0))
        _matched = sum(1 for k in labels_map if k in cam["file"].values)

        print(f"  LGN=1  : {n_lgn_csv} files")
        print(f"  LGN=0  : {n_nonlgn_csv} files")
        print(f"  Matched in features.csv : {_matched} / {len(labels_map)}\n")

        if _matched == 0:
            print("  ✗ 0 files matched. Showing first 3 from each side:")
            print("  manual_labels.csv entries:")
            for k in list(labels_map.keys())[:3]:
                print(f"    '{k}'")
            print("  features.csv 'file' column (first 3):")
            for v in cam["file"].head(3).values:
                print(f"    '{v}'")
            sys.exit(1)

        # Match on cleaned filename (strip extension, strip whitespace)
        cam["_file_clean"] = (cam["file"].astype(str)
                               .str.replace(r"\.ns6$", "", regex=True)
                               .str.strip())
        cam["_label"] = cam["_file_clean"].map(labels_map)
        labelled_df   = cam[cam["_label"].notna()].copy()
        unlabelled_df = cam[cam["_label"].isna()].copy()
        labelled_df["is_lgn"] = labelled_df["_label"].astype(int)

        n_lgn_lab    = int((labelled_df["is_lgn"] == 1).sum())
        n_nonlgn_lab = int((labelled_df["is_lgn"] == 0).sum())
        n_files_lgn  = int((labelled_df[labelled_df["is_lgn"]==1]
                            ["file"].nunique()))
        n_files_non  = int((labelled_df[labelled_df["is_lgn"]==0]
                            ["file"].nunique()))

        print(f"  Labelled   : {len(labelled_df)} units  "
              f"({n_lgn_lab} LGN / {n_nonlgn_lab} non-LGN)  "
              f"[{n_files_lgn} LGN files / {n_files_non} non-LGN files]")
        print(f"  Unlabelled : {len(unlabelled_df)} units  "
              f"← prediction targets\n")

        if n_lgn_lab < 6 or n_nonlgn_lab < 6:
            print(f"  ⚠ Too few labelled units per class "
                  f"(need ≥6 each, have {n_lgn_lab} LGN / {n_nonlgn_lab} non-LGN).")
            print("  Strategy B skipped. Add more rows to manual_labels.csv.\n")
        else:
            # ── Leave-One-File-Out CV ──────────────────────────────
            print("  Running Leave-One-File-Out CV …")
            y_true, y_pred_cv, y_prob_cv = lofo_cv(
                labelled_df.reset_index(drop=True), WF_FEATURES_SUPER
            )

            if len(y_true) == 0:
                print("  ⚠ LOFO-CV returned no predictions "
                      "(some files may have only one class in training).")
            else:
                n_lgn_eval    = int(y_true.sum())
                n_nonlgn_eval = int((y_true == 0).sum())

                lofo_auc, sens, spec, acc = plot_confusion_and_roc(
                    y_true, y_pred_cv, y_prob_cv,
                    n_lgn_eval, n_nonlgn_eval,
                    os.path.join(RESULTS_DIR, "confusion_roc_b.png")
                )

                print(f"\n  LOFO-CV Results ({len(y_true)} evaluated units):")
                print(f"    AUC         = {lofo_auc:.3f}")
                print(f"    Accuracy    = {acc:.3f}")
                print(f"    Sensitivity = {sens:.3f}  (true LGN correctly called LGN)")
                print(f"    Specificity = {spec:.3f}  (true non-LGN correctly called non-LGN)")
                print()
                print(classification_report(y_true, y_pred_cv,
                                            target_names=["non-LGN", "LGN"],
                                            zero_division=0))

                # Interpretation
                if lofo_auc >= 0.85:
                    interp = "✓ Good — predictions on unlabelled cells are reliable."
                elif lofo_auc >= 0.70:
                    interp = ("⚠ Acceptable — predictions are usable but "
                              "borderline cells may be misclassified. "
                              "Adding more labels will help.")
                else:
                    interp = ("✗ Poor — waveform features alone may not "
                              "separate LGN from non-LGN in your data. "
                              "Add more labels or check feature quality.")
                print(f"  Interpretation: {interp}\n")

            # ── Retrain on all labelled, predict unlabelled ────────
            X_lab, sc_b   = impute_scale(labelled_df,   WF_FEATURES_SUPER)
            y_lab         = labelled_df["is_lgn"].values.astype(int)
            X_cam_b, sc_b = impute_scale(cam,           WF_FEATURES_SUPER, scaler=sc_b)

            svm_b = SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42)
            svm_b.fit(X_lab, y_lab)

            strategy_b_scores = svm_b.decision_function(X_cam_b)
            strategy_b_pred   = svm_b.predict(X_cam_b)

            if len(unlabelled_df) > 0:
                X_ul, _        = impute_scale(unlabelled_df, WF_FEATURES_SUPER, scaler=sc_b)
                unlab_proba_arr = svm_b.predict_proba(X_ul)[:, 1]
                unlabelled_df  = unlabelled_df.copy()
                unlabelled_df["lgn_prob_b"] = unlab_proba_arr

            n_b = int((strategy_b_pred == 1).sum())
            print(f"  Full-dataset prediction: {n_b} LGN  |  "
                  f"{len(strategy_b_pred)-n_b} non-LGN\n")

    # ════════════════════════════════════════════════════════════════
    #  COMBINE + FINAL LABELS
    # ════════════════════════════════════════════════════════════════
    print("═"*55)
    print("FINAL CLASSIFICATION")
    print("═"*55 + "\n")

    cam_out = cam.drop(columns=["_label", "_file_clean"], errors="ignore").copy()

    if strategy_a_scores is not None:
        cam_out["score_a"] = strategy_a_scores
        cam_out["pred_a"]  = strategy_a_pred

    if strategy_b_scores is not None:
        cam_out["score_b"]    = strategy_b_scores
        cam_out["pred_b"]     = (strategy_b_pred == 1).astype(int)

    # Preserve manual labels in output
    if not labelled_df.empty:
        cam_out["manual_label"] = cam_out["file"].map(
            labelled_df.set_index("file")["is_lgn"].to_dict()
        )

    # Final decision
    if strategy_a_scores is not None and strategy_b_scores is not None:
        s_a = (strategy_a_scores - strategy_a_scores.mean()) / \
               (strategy_a_scores.std() + 1e-9)
        s_b = (strategy_b_scores - strategy_b_scores.mean()) / \
               (strategy_b_scores.std() + 1e-9)
        # Weight B more heavily (within-dataset, better matched)
        # Increase B weight further if its AUC was high
        w_b = 0.70 if lofo_auc >= 0.80 else 0.60
        combined = (1 - w_b) * s_a + w_b * s_b
        cam_out["score_combined"] = combined
        cam_out["final_lgn"]      = (combined > 0).astype(int)
        print(f"  Using {int((1-w_b)*100)}% Strategy A + {int(w_b*100)}% "
              f"Strategy B  (AUC={lofo_auc:.3f})")
    elif strategy_a_scores is not None:
        cam_out["score_combined"] = strategy_a_scores
        cam_out["final_lgn"]      = strategy_a_pred
        print("  Using Strategy A only (no manual labels)")
    elif strategy_b_scores is not None:
        cam_out["score_combined"] = strategy_b_scores
        cam_out["final_lgn"]      = (strategy_b_pred == 1).astype(int)
        print("  Using Strategy B only")
    else:
        print("  ✗ No strategy produced results. Check paths and inputs.")
        sys.exit(1)

    if "manual_label" in cam_out.columns:
        known = cam_out["manual_label"].notna()
        cam_out.loc[known, "final_lgn"] = \
            cam_out.loc[known, "manual_label"].astype(int)
        print(f"  Manual labels override model for {known.sum()} units")

    n_lgn_f = int(cam_out["final_lgn"].sum())
    n_tot   = len(cam_out)
    print(f"\n  Final totals:")
    print(f"    LGN     : {n_lgn_f:4d}  ({100*n_lgn_f/n_tot:.1f}%)")
    print(f"    non-LGN : {n_tot-n_lgn_f:4d}  ({100*(n_tot-n_lgn_f)/n_tot:.1f}%)")

    print("\n  Per-session breakdown:")
    print(f"  {'Session':<14}  {'Files':>5}  {'Total':>6}  "
          f"{'LGN':>5}  {'non-LGN':>8}  {'%LGN':>6}")
    print(f"  {'─'*14}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*6}")
    for sess, grp in cam_out.groupby("session"):
        nf  = grp["file"].nunique()
        n   = len(grp)
        nl  = int(grp["final_lgn"].sum())
        pct = 100 * nl / n if n > 0 else 0
        print(f"  {sess:<14}  {nf:>5}  {n:>6}  {nl:>5}  "
              f"{n-nl:>8}  {pct:>5.1f}%")

    # ── Save outputs ───────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS_DIR, "features_lgn_labelled.csv")
    cam_out.to_csv(out_csv,               index=False)
    cam_out.to_csv(CAMELOT_FEATURES_CSV,  index=False)
    print(f"\n  Saved → {out_csv}")
    print(f"  Updated → {CAMELOT_FEATURES_CSV}")

    # ── Plots ──────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # PCA of waveform features
    if strategy_a_scores is not None and not ref_df.empty:
        X_ref_pca, sc_pca = impute_scale(ref_df,    WF_FEATURES_CROSS)
        X_cam_pca, sc_pca = impute_scale(cam_out,   WF_FEATURES_CROSS, scaler=sc_pca)

        pca_fit = PCA(n_components=2, random_state=42)
        pca_fit.fit(np.vstack([X_ref_pca, X_cam_pca]))

        X_lab_lgn = np.zeros((0, len(WF_FEATURES)))
        X_lab_non = np.zeros((0, len(WF_FEATURES)))
        if not labelled_df.empty:
            ll = labelled_df[labelled_df["is_lgn"]==1]
            ln = labelled_df[labelled_df["is_lgn"]==0]
            if len(ll): X_lab_lgn, _ = impute_scale(ll, WF_FEATURES_CROSS, scaler=sc_pca)
            if len(ln): X_lab_non, _ = impute_scale(ln, WF_FEATURES_CROSS, scaler=sc_pca)

        X_ul = np.zeros((0, len(WF_FEATURES)))
        unlab_p = np.array([])
        if len(unlabelled_df):
            X_ul, _  = impute_scale(unlabelled_df, WF_FEATURES_CROSS, scaler=sc_pca)
            if "lgn_prob_b" in unlabelled_df.columns:
                unlab_p = unlabelled_df["lgn_prob_b"].values
            elif strategy_a_scores is not None:
                # Normalise OC scores 0-1 as proxy
                sc_vals = strategy_a_scores[cam_out["file"].isin(
                    unlabelled_df["file"])]
                mn, mx = sc_vals.min(), sc_vals.max()
                unlab_p = (sc_vals - mn) / (mx - mn + 1e-9)
            else:
                unlab_p = np.full(len(unlabelled_df), 0.5)

        plot_wf_pca(X_ref_pca, X_lab_lgn, X_lab_non, X_ul, unlab_p,
                    pca_fit,
                    os.path.join(RESULTS_DIR, "pca_waveform_features.png"))

        plot_feature_boxes(
            ref_df, cam_out,
            os.path.join(RESULTS_DIR, "feature_comparison.png")
        )

    plot_session_summary(
        cam_out,
        os.path.join(RESULTS_DIR, "session_summary.png")
    )

    print(f"\n  ✓ All done. Results in:\n    {RESULTS_DIR}\n")
    print("  Output files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        sz = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1e3
        print(f"    {f:<50}  {sz:6.1f} kB")


if __name__ == "__main__":
    main()
