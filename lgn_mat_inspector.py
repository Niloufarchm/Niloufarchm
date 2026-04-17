"""
Step 1 — LGN Reference Dataset Inspector
==========================================
Run this FIRST before lgn_vs_nonlgn_classifier.py.

It scans every .mat file in the reference directory and prints:
  • All top-level variable names and nested struct fields
  • Shape / dtype of each variable
  • A preview of the first numeric values

Usage
─────
    python lgn_mat_inspector.py

Output
──────
    Prints a structured report to the terminal.
    Also saves  lgn_mat_field_report.txt  next to this script.
"""

import os
import sys
import numpy as np
import scipy.io

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("  ⚠  h5py not installed — install with:  pip install h5py")

# ──────────────────────────────────────────────────────────────────────
REFERENCE_DIR = (
    "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/"
    "Monkey/Rapid adaptation of primate LGN neurons to drifting grating stimulation/"
    "10.12751_g-node.kvut7v/data"
)
MAX_FILES_TO_INSPECT = 3   # first N non-ghost files is usually enough
MAX_STRUCT_DEPTH     = 4   # how deep to recurse into nested structs
# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
#  IMPORTANT: summarise_value must be defined before describe_struct
# ══════════════════════════════════════════════════════════════════════

def summarise_value(v) -> str:
    """Return a human-readable one-liner for any MATLAB-loaded value."""
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        preview = ""
        if flat.size > 0 and np.issubdtype(flat.dtype, np.number):
            sample = flat[:min(6, flat.size)]
            preview = f"  first values: {np.round(sample, 4)}"
        return f"ndarray  shape={v.shape}  dtype={v.dtype}{preview}"
    elif isinstance(v, (int, float, complex, np.generic)):
        return f"scalar   value={v}"
    elif isinstance(v, str):
        return f"string   '{v[:100]}'"
    elif isinstance(v, bytes):
        return f"bytes    len={len(v)}"
    elif isinstance(v, np.void):
        return f"struct-element  fields={v.dtype.names}"
    elif hasattr(v, "_fieldnames"):
        return f"[MATLAB struct]  fields={v._fieldnames}"
    else:
        return f"{type(v).__name__}"


def describe_struct(obj, depth: int = 0, max_depth: int = MAX_STRUCT_DEPTH) -> list:
    """Recursively describe a scipy mat_struct (summarise_value must be defined first)."""
    lines  = []
    indent = "  " * depth

    if not hasattr(obj, "_fieldnames"):
        return [f"{indent}{summarise_value(obj)}"]

    for field in obj._fieldnames:
        try:
            val = getattr(obj, field)
        except Exception as e:
            lines.append(f"{indent}{field}:  <error: {e}>")
            continue

        if hasattr(val, "_fieldnames") and depth < max_depth:
            lines.append(f"{indent}{field}:  [struct]  subfields={val._fieldnames}")
            lines.extend(describe_struct(val, depth + 1, max_depth))

        elif isinstance(val, np.ndarray) and val.dtype == object and val.size > 0:
            lines.append(f"{indent}{field}:  cell-array  shape={val.shape}")
            elem = val.flat[0]
            if hasattr(elem, "_fieldnames"):
                lines.append(f"{indent}  [0]:  [struct]  subfields={elem._fieldnames}")
                lines.extend(describe_struct(elem, depth + 2, max_depth))
            else:
                lines.append(f"{indent}  [0]:  {summarise_value(elem)}")

        elif isinstance(val, np.ndarray) and val.dtype.names:
            lines.append(f"{indent}{field}:  structured-array  "
                         f"shape={val.shape}  fields={val.dtype.names}")
        else:
            lines.append(f"{indent}{field}:  {summarise_value(val)}")

    return lines


def decode_filename(fname: str) -> str:
    """Parse the Daumail 2023 filename convention."""
    base   = os.path.splitext(os.path.basename(fname))[0]
    parts  = base.split("_")
    labels = [
        "date",
        "eye (I=ipsi / C=contra)",
        "penetration",
        "unit/cluster",
        "condition",
        "stability",
        "analysis",
        "data_type",
    ]
    lines = [f"  Filename anatomy: {base}"]
    for label, part in zip(labels, parts):
        lines.append(f"    {label:<28} = {part}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  FILE LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_scipy(path: str) -> dict:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def load_h5py(path: str) -> dict:
    result = {}
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = np.array(obj)
        f.visititems(visitor)
    return result


def load_mat(path: str) -> tuple:
    """Returns (data_dict, loader_name)."""
    try:
        d = load_scipy(path)
        if d:
            return d, "scipy"
    except Exception:
        pass
    if HAS_H5PY:
        try:
            d = load_h5py(path)
            if d:
                return d, "h5py"
        except Exception as e:
            return {}, f"h5py-error: {e}"
    return {}, "failed"


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    if not os.path.isdir(REFERENCE_DIR):
        print(f"✗ Directory not found:\n  {REFERENCE_DIR}")
        sys.exit(1)

    # Collect mat files — skip macOS ghost companions (._*)
    mat_files = sorted(
        f for f in os.listdir(REFERENCE_DIR)
        if f.endswith(".mat") and not f.startswith("._")
    )
    for entry in sorted(os.listdir(REFERENCE_DIR)):
        sub = os.path.join(REFERENCE_DIR, entry)
        if os.path.isdir(sub):
            for f in sorted(os.listdir(sub)):
                if f.endswith(".mat") and not f.startswith("._"):
                    mat_files.append(os.path.join(entry, f))

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  LGN Reference Dataset — Structure Report   ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"\nDirectory  : {REFERENCE_DIR}")
    print(f"Valid .mat files (ghost files excluded): {len(mat_files)}")
    print(f"Inspecting first {MAX_FILES_TO_INSPECT} …\n")

    if not mat_files:
        print("  ✗ No .mat files found.")
        sys.exit(1)

    report_lines = [
        "LGN Reference Dataset — Field Report",
        f"Directory: {REFERENCE_DIR}",
        f"Valid .mat files: {len(mat_files)}",
        "",
    ]

    for i, rel_path in enumerate(mat_files[:MAX_FILES_TO_INSPECT]):
        full_path = os.path.join(REFERENCE_DIR, rel_path)
        size_mb   = os.path.getsize(full_path) / 1e6

        sep    = f"{'─'*65}"
        header = f"File {i+1}: {rel_path}  ({size_mb:.2f} MB)"
        print(sep); print(header)
        report_lines += [sep, header]

        anatomy = decode_filename(rel_path)
        print(anatomy)
        report_lines.append(anatomy)

        data, loader = load_mat(full_path)
        lline = f"  Loader: {loader}"
        print(lline); report_lines.append(lline)

        if not data:
            msg = "  ✗ Could not read file."
            print(msg); report_lines.append(msg)
            continue

        kline = f"\n  Top-level keys ({len(data)}): {list(data.keys())}\n"
        print(kline); report_lines.append(kline)

        for key, val in sorted(data.items()):

            # MATLAB struct → recurse fully
            if hasattr(val, "_fieldnames"):
                hdr = f"  ▶ {key}:  [struct]  fields = {val._fieldnames}"
                print(hdr); report_lines.append(hdr)
                for sl in describe_struct(val, depth=2):
                    print(sl); report_lines.append(sl)

            # Cell array
            elif isinstance(val, np.ndarray) and val.dtype == object:
                hdr = f"  ▶ {key}:  cell-array  shape={val.shape}"
                print(hdr); report_lines.append(hdr)
                for ci in range(min(2, val.size)):
                    elem  = val.flat[ci]
                    eline = f"    [{ci}]: {summarise_value(elem)}"
                    print(eline); report_lines.append(eline)
                    if hasattr(elem, "_fieldnames"):
                        for sl in describe_struct(elem, depth=3):
                            print(sl); report_lines.append(sl)

            # Structured numpy array
            elif isinstance(val, np.ndarray) and val.dtype.names:
                hdr = (f"  ▶ {key}:  structured-array  shape={val.shape}  "
                       f"fields={val.dtype.names}")
                print(hdr); report_lines.append(hdr)

            # Plain array / scalar
            else:
                hdr = f"  ▶ {key}:  {summarise_value(val)}"
                print(hdr); report_lines.append(hdr)

        print(); report_lines.append("")

    # Save report
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lgn_mat_field_report.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(report_lines))

    print(f"\n✓ Report saved → {out_path}")
    print("\nNext step:")
    print("  Open lgn_vs_nonlgn_classifier.py, set the FIELD_* constants")
    print("  to match the field names printed above, then run the classifier.\n")


if __name__ == "__main__":
    main()