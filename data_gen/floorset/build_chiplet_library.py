# --- repo bootstrap -------------------------------------------------------
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "solo2stack")):
        _sys.path.insert(0, _d)
        break
    _d = _os.path.dirname(_d)
from solo2stack.paths import P, repo_path  # noqa: E402

# FloorSet itself is not vendored here: the dataset and the iccad2026contest
# package are downloaded separately (see data_gen/floorset/README.md) and
# located through floorset_root in configs/paths.yaml.
_sys.path.insert(0, P["floorset_root"])
# --------------------------------------------------------------------------
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from iccad2026contest.iccad2026_evaluate import get_training_dataloader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# Utilities
# -----------------------------
def find_sol_tensor(obj):
    """
    Recursively search for the floorplan solution tensor.
    We want a tensor whose last dimension is 4:
      [w, h, x, y] per block.
    """
    if torch.is_tensor(obj):
        if obj.ndim >= 2 and obj.shape[-1] == 4:
            return obj
        return None

    if isinstance(obj, dict):
        for v in obj.values():
            found = find_sol_tensor(v)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_sol_tensor(item)
            if found is not None:
                return found
        return None

    return None


def unwrap_batch(batch):
    sol = find_sol_tensor(batch)

    if sol is None:
        print("Could not find a tensor with last dimension = 4.")
        print("Top-level batch type:", type(batch))

        if isinstance(batch, (list, tuple)):
            print("Top-level batch length:", len(batch))
            for i, item in enumerate(batch):
                if torch.is_tensor(item):
                    print(f"  item[{i}]: tensor shape={tuple(item.shape)}")
                else:
                    print(f"  item[{i}]: type={type(item)}")
        elif isinstance(batch, dict):
            print("Top-level dict keys:", list(batch.keys()))
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"  key={k}: tensor shape={tuple(v.shape)}")
                else:
                    print(f"  key={k}: type={type(v)}")

        raise ValueError("No valid sol tensor found in batch.")

    sol = sol.detach().cpu()

    # Common shapes:
    #   (n_blocks, 4)
    #   (1, n_blocks, 4)
    #   (1, 1, n_blocks, 4)
    #   flattened but divisible by 4
    if sol.ndim == 3 and sol.shape[0] == 1 and sol.shape[-1] == 4:
        sol = sol[0]
    elif sol.ndim == 4 and sol.shape[0] == 1 and sol.shape[1] == 1 and sol.shape[-1] == 4:
        sol = sol[0, 0]
    elif sol.ndim == 2 and sol.shape[1] == 4:
        pass
    elif sol.ndim == 1 and sol.numel() % 4 == 0:
        sol = sol.view(-1, 4)
    else:
        raise ValueError(f"Unexpected sol shape: {tuple(sol.shape)}")

    return {"sol": sol}


def sol_to_df(sol):
    sol = np.asarray(sol)

    if sol.ndim == 3 and sol.shape[0] == 1 and sol.shape[-1] == 4:
        sol = sol[0]

    if sol.ndim == 2 and sol.shape[1] == 4:
        blocks = sol
    elif sol.ndim == 1 and sol.size % 4 == 0:
        blocks = sol.reshape(-1, 4)
    elif sol.ndim == 2 and sol.shape[0] == 1 and sol.shape[1] % 4 == 0:
        blocks = sol.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected sol shape in sol_to_df: {sol.shape}")

    rows = []
    for i, row in enumerate(blocks):
        w, h, x, y = map(float, row[:4])
        rows.append({
            "UnitName": f"B{i}",
            "X": x,
            "Y": y,
            "Length_m": w,
            "Width_m": h,
            "ConfigFile": "",
            "Label": "Si",
        })

    return pd.DataFrame(rows)


def scale_to_2mm_full(df, target_size_m=0.002):
    """
    Non-uniform scaling to exactly fill 2mm x 2mm.
    No grid snapping.
    """
    df = df.copy()

    x_min = df["X"].min()
    y_min = df["Y"].min()
    x_max = (df["X"] + df["Length_m"]).max()
    y_max = (df["Y"] + df["Width_m"]).max()

    orig_w = x_max - x_min
    orig_h = y_max - y_min

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid floorplan bounding box.")

    scale_x = target_size_m / orig_w
    scale_y = target_size_m / orig_h

    df["X"] = (df["X"] - x_min) * scale_x
    df["Y"] = (df["Y"] - y_min) * scale_y
    df["Length_m"] = df["Length_m"] * scale_x
    df["Width_m"] = df["Width_m"] * scale_y

    df["X"] = df["X"].clip(0.0, target_size_m)
    df["Y"] = df["Y"].clip(0.0, target_size_m)

    return df, {
        "orig_w_m": float(orig_w),
        "orig_h_m": float(orig_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "target_size_m": float(target_size_m),
    }


def synthesize_power_trace_area_weighted_gaussian(
    df,
    seed=0,
    total_power_w=4.0,
    cv=0.3,
    density_floor_ratio=0.05,
):
    """
    Area-weighted power assignment with Gaussian density variation.

    Method:
      1) Split total power according to block area ratio.
      2) Perturb each block with a Gaussian factor N(1, cv).
      3) Clip factors to stay positive.
      4) Renormalize so the sum is exactly total_power_w.

    This keeps the expected power proportional to area, while the
    power-density variation across blocks follows a normal distribution.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    if n == 0:
        raise ValueError("Empty floorplan dataframe.")

    areas_m2 = (out["Length_m"].to_numpy() * out["Width_m"].to_numpy()).astype(np.float64)
    if np.any(areas_m2 <= 0):
        raise ValueError("Non-positive block area detected.")

    total_area_m2 = float(areas_m2.sum())
    if total_area_m2 <= 0:
        raise ValueError("Invalid total area.")

    total_power_w = float(total_power_w)
    mean_density_w_m2 = total_power_w / total_area_m2
    sigma_density_w_m2 = cv * mean_density_w_m2

    area_ratio = areas_m2 / total_area_m2
    base_power_w = total_power_w * area_ratio

    # Gaussian perturbation around 1.0 with the requested CV.
    # Clip only to keep powers positive; the renormalization preserves total power.
    factors = rng.normal(loc=1.0, scale=cv, size=n)
    factors = np.clip(factors, density_floor_ratio, None)

    power_w = base_power_w * factors
    power_w_sum = float(power_w.sum())
    if power_w_sum <= 0:
        raise ValueError("Power synthesis failed: non-positive total power before normalization.")

    power_w *= total_power_w / power_w_sum
    power_density_w_m2 = power_w / areas_m2

    out_power = pd.DataFrame({
        "UnitName": out["UnitName"].astype(str),
        "Power": power_w.astype(float),
    })

    stats = {
        "total_power_w": float(total_power_w),
        "total_area_m2": float(total_area_m2),
        "mean_density_target_w_m2": float(mean_density_w_m2),
        "sigma_density_target_w_m2": float(sigma_density_w_m2),
        "cv": float(cv),
        "density_floor_ratio": float(density_floor_ratio),
        "realized_power_sum_w": float(power_w.sum()),
        "realized_density_mean_w_m2": float(np.mean(power_density_w_m2)),
        "realized_density_std_w_m2": float(np.std(power_density_w_m2)),
        "realized_density_min_w_m2": float(np.min(power_density_w_m2)),
        "realized_density_max_w_m2": float(np.max(power_density_w_m2)),
    }

    return out_power, stats


def build_power_density_map(df, ptr_df, grid_size=400):
    """
    Build a 2D power density map using grid_size x grid_size.
    """
    merged = df.merge(ptr_df, on="UnitName", how="left")
    merged["Power"] = merged["Power"].fillna(0.0)

    x_min = float(df["X"].min())
    y_min = float(df["Y"].min())
    x_max = float((df["X"] + df["Length_m"]).max())
    y_max = float((df["Y"] + df["Width_m"]).max())

    chip_w = x_max - x_min
    chip_h = y_max - y_min

    if chip_w <= 0 or chip_h <= 0:
        raise ValueError("Invalid chip extent for powermap.")

    def x_to_px(x_m: float) -> int:
        return int(np.floor((x_m - x_min) / chip_w * grid_size))

    def y_to_py(y_m: float) -> int:
        return int(np.floor((y_m - y_min) / chip_h * grid_size))

    power_density_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _, r in merged.iterrows():
        x0 = x_to_px(float(r["X"]))
        y0 = y_to_py(float(r["Y"]))
        x1 = x_to_px(float(r["X"] + r["Length_m"]))
        y1 = y_to_py(float(r["Y"] + r["Width_m"]))

        x0, x1 = (x0, x1) if x1 >= x0 else (x1, x0)
        y0, y1 = (y0, y1) if y1 >= y0 else (y1, y0)

        x0 = max(0, min(grid_size, x0))
        x1 = max(0, min(grid_size, x1))
        y0 = max(0, min(grid_size, y0))
        y1 = max(0, min(grid_size, y1))

        if x1 <= x0 or y1 <= y0:
            continue

        power_w = float(r["Power"])
        area_m2 = float(r["Length_m"] * r["Width_m"])
        density = 0.0 if area_m2 <= 0 else (power_w / area_m2)
        power_density_map[y0:y1, x0:x1] += density

    return power_density_map


def save_floorplan_csv(df, out_csv):
    df_out = df.rename(columns={
        "Length_m": "Length (m)",
        "Width_m": "Width (m)",
    })[["UnitName", "X", "Y", "Length (m)", "Width (m)", "ConfigFile", "Label"]]
    df_out.to_csv(out_csv, index=False, float_format="%.10f")


def save_ptrace_csv(ptr_df, out_csv):
    ptr_df.to_csv(out_csv, index=False, float_format="%.10f")


def save_lcf_csv(flp_csv, ptr_csv, out_csv, thickness=0.0005):
    """
    Generate single-layer lcf.csv for PACT.
    Use absolute paths to avoid cwd issues.
    """
    df = pd.DataFrame([{
        "Layer": 0,
        "FloorplanFile": str(Path(flp_csv).resolve()),
        "Thickness (m)": thickness,
        "PtraceFile": str(Path(ptr_csv).resolve()),
        "LateralHeatFlow": True,
    }])
    df.to_csv(out_csv, index=False)


def save_floorplan_png(df, out_png, target_size_m=0.002, draw_labels=False):
    if not HAS_MPL:
        return

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    for _, r in df.iterrows():
        x0 = float(r["X"])
        y0 = float(r["Y"])
        x1 = x0 + float(r["Length_m"])
        y1 = y0 + float(r["Width_m"])

        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1)
        ax.add_patch(rect)

        if draw_labels:
            cx = x0 + 0.5 * (x1 - x0)
            cy = y0 + 0.5 * (y1 - y0)
            ax.text(cx, cy, str(r["UnitName"]), ha="center", va="center", fontsize=6)

    ax.set_xlim(0, target_size_m)
    ax.set_ylim(0, target_size_m)
    ax.set_title("Floorplan (scaled to 2mm x 2mm)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_powermap_png(power_map, out_png):
    if not HAS_MPL:
        return

    plt.figure(figsize=(7, 7))
    plt.imshow(power_map, origin="lower", cmap="jet")
    plt.colorbar(label="Power Density (W/m^2)")
    plt.title("Power Density Map (400x400)")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_group_json(chiplet_names, num_groups, group_size, out_json, seed=0):
    rng = np.random.default_rng(seed)
    names = list(chiplet_names)
    if len(names) < num_groups * group_size:
        raise ValueError(f"Need at least {num_groups * group_size} chiplets, but got {len(names)}")

    rng.shuffle(names)
    selected = names[:num_groups * group_size]
    groups = np.array_split(selected, num_groups)

    group_dict = {f"group_{i+1:02d}": [str(x) for x in g.tolist()] for i, g in enumerate(groups)}
    with open(out_json, "w") as f:
        json.dump(group_dict, f, indent=2)
    return group_dict


def write_manifest(records, out_json):
    with open(out_json, "w") as f:
        json.dump({"chiplets": records}, f, indent=2)


def write_pact_run_template(records, out_sh, config_file, model_params_file):
    """
    Template only. Edit BC_CONFIGS inside the script to point to your 5 BC configs.
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append('PACT_PY="${PACT_PY:-../src/PACT.py}"')
    lines.append(f'CONFIG_FILE="${{CONFIG_FILE:-{config_file}}}"')
    lines.append(f'MODEL_PARAMS_FILE="${{MODEL_PARAMS_FILE:-{model_params_file}}}"')
    lines.append('OUT_ROOT="${OUT_ROOT:-pact_runs}"')
    lines.append("")
    lines.append("# Edit these to your 5 boundary-condition config files.")
    lines.append("BC_CONFIGS=(")
    lines.append('  "boundary_conditions/bc_01.config"')
    lines.append('  "boundary_conditions/bc_02.config"')
    lines.append('  "boundary_conditions/bc_03.config"')
    lines.append('  "boundary_conditions/bc_04.config"')
    lines.append('  "boundary_conditions/bc_05.config"')
    lines.append(")")
    lines.append("")
    lines.append('mkdir -p "$OUT_ROOT"')
    lines.append("")

    for rec in records:
        chip_id = rec["chiplet_id"]
        lcf = rec["lcf_csv"]
        lines.append(f'mkdir -p "$OUT_ROOT/{chip_id}"')
        lines.append('for bc in "${BC_CONFIGS[@]}"; do')
        lines.append('  bc_name=$(basename "$bc" .config)')
        lines.append(f'  out_dir="$OUT_ROOT/{chip_id}/${{bc_name}}"')
        lines.append('  mkdir -p "$out_dir"')
        lines.append(f'  python "$PACT_PY" "{lcf}" "$bc" "$MODEL_PARAMS_FILE" --gridSteadyFile "$out_dir/{chip_id}_${{bc_name}}_steady"')
        lines.append('done')
        lines.append("")

    with open(out_sh, "w") as f:
        f.write("\n".join(lines))
    os.chmod(out_sh, 0o755)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--target-mm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--power-mode",
        type=str,
        default="area_weighted_gaussian",
        choices=["area_weighted_gaussian", "uniform"],
    )
    parser.add_argument("--cv", type=float, default=0.3)
    parser.add_argument("--total-power-w", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--thickness-m", type=float, default=0.0005)
    parser.add_argument("--save-pngs", action="store_true", default=True)
    parser.add_argument("--out-root", type=str, default=P["chiplet_library"],
                        help="where the chiplet library is written "
                             "(chiplet_library in configs/paths.yaml)")
    parser.add_argument("--config-file", type=str,
                        default=repo_path("configs", "pact", "bc_4.config"))
    parser.add_argument("--model-params-file", type=str,
                        default=repo_path("configs", "pact", "modelParam.config"))
    args = parser.parse_args()

    target_size_m = args.target_mm / 1000.0
    grid_size = args.grid_size

    out_root = Path(args.out_root)
    raw_lcf_dir = out_root / "raw_lcf"
    raw_flp_dir = out_root / "raw_floorplans"
    raw_ptr_dir = out_root / "raw_ptraces"
    png_flp_dir = out_root / "floorplan_png"
    png_pow_dir = out_root / "powermap_png"
    npy_pow_dir = out_root / "powermap_npy"
    meta_dir = out_root / "meta"
    group_dir = out_root / "groups"
    run_dir = out_root / "pact_jobs"

    for d in [raw_lcf_dir, raw_flp_dir, raw_ptr_dir, png_flp_dir, png_pow_dir, npy_pow_dir, meta_dir, group_dir, run_dir]:
        d.mkdir(parents=True, exist_ok=True)

    loader = get_training_dataloader(batch_size=1, num_samples=args.num_samples)

    records = []
    chiplet_names = []

    for idx, batch in enumerate(loader):
        sample = unwrap_batch(batch)
        if "sol" not in sample:
            raise KeyError(f"Sample {idx} has no 'sol' key")

        sol = sample["sol"]
        if torch.is_tensor(sol):
            sol = sol.numpy()
        if sol.ndim == 3:
            sol = sol[0]
        sol = np.asarray(sol, dtype=np.float32)

        chiplet_id = f"chiplet{idx:03d}"
        chiplet_names.append(chiplet_id)

        df = sol_to_df(sol)
        df_pact, scale_meta = scale_to_2mm_full(df, target_size_m=target_size_m)

        x2_max = float((df_pact["X"] + df_pact["Length_m"]).max())
        y2_max = float((df_pact["Y"] + df_pact["Width_m"]).max())
        if x2_max > target_size_m + 1e-12 or y2_max > target_size_m + 1e-12:
            raise ValueError(f"{chiplet_id}: floorplan exceeds die boundary.")

        ptr_df, power_stats = synthesize_power_trace_area_weighted_gaussian(
            df_pact,
            seed=args.seed + idx,
            total_power_w=args.total_power_w,
            cv=args.cv,
        )

        power_density_map = build_power_density_map(df_pact, ptr_df, grid_size=grid_size)

        flp_csv = raw_flp_dir / f"{chiplet_id}_flp.csv"
        ptr_csv = raw_ptr_dir / f"{chiplet_id}_ptrace.csv"
        lcf_csv = raw_lcf_dir / f"{chiplet_id}_lcf.csv"
        powermap_npy = npy_pow_dir / f"{chiplet_id}_powermap.npy"

        save_floorplan_csv(df_pact, flp_csv)
        save_ptrace_csv(ptr_df, ptr_csv)
        save_lcf_csv(flp_csv, ptr_csv, lcf_csv, thickness=args.thickness_m)
        np.save(powermap_npy, power_density_map)

        if args.save_pngs:
            save_floorplan_png(df_pact, png_flp_dir / f"{chiplet_id}_floorplan.png", target_size_m=target_size_m)
            save_powermap_png(power_density_map, png_pow_dir / f"{chiplet_id}_powermap.png")

        meta = {
            "chiplet_id": chiplet_id,
            "sample_index": idx,
            "n_blocks": int(sol.shape[0]),
            "target_size_m": target_size_m,
            "grid_size": grid_size,
            "scale_meta": scale_meta,
            "power_assignment": {
                "mode": args.power_mode,
                "total_power_w": args.total_power_w,
                "cv": args.cv,
                "stats": power_stats,
            },
            "floorplan_csv": str(flp_csv),
            "ptrace_csv": str(ptr_csv),
            "lcf_csv": str(lcf_csv),
            "powermap_npy": str(powermap_npy),
        }
        with open(meta_dir / f"{chiplet_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        records.append({
            "chiplet_id": chiplet_id,
            "group": None,
            "n_blocks": int(sol.shape[0]),
            "flp_csv": str(flp_csv),
            "ptrace_csv": str(ptr_csv),
            "lcf_csv": str(lcf_csv),
            "powermap_npy": str(powermap_npy),
            "meta_json": str(meta_dir / f"{chiplet_id}.json"),
        })

        print(
            f"[{idx+1}/{args.num_samples}] saved {chiplet_id} with {sol.shape[0]} blocks | "
            f"P_total={power_stats['realized_power_sum_w']:.4f} W | "
            f"mean_d={power_stats['realized_density_mean_w_m2']:.3e} W/m^2 | "
            f"std_d={power_stats['realized_density_std_w_m2']:.3e} W/m^2"
        )

    group_dict = write_group_json(
        chiplet_names=chiplet_names,
        num_groups=args.num_groups,
        group_size=args.group_size,
        out_json=group_dir / "chiplet_groups.json",
        seed=args.seed,
    )

    name_to_group = {}
    for gname, names in group_dict.items():
        for n in names:
            name_to_group[n] = gname
    for rec in records:
        rec["group"] = name_to_group.get(rec["chiplet_id"], None)

    write_manifest(records, out_root / "chiplet_manifest.json")
    write_pact_run_template(records, run_dir / "run_pact.sh", args.config_file, args.model_params_file)

    print("")
    print("Pipeline finished.")
    print(f"Root:       {out_root}")
    print(f"Floorplans: {raw_flp_dir}")
    print(f"Ptraces:    {raw_ptr_dir}")
    print(f"LCFs:       {raw_lcf_dir}")
    print(f"Powermaps:  {npy_pow_dir}")
    print(f"Groups:     {group_dir / 'chiplet_groups.json'}")
    print(f"Manifest:   {out_root / 'chiplet_manifest.json'}")
    print(f"PACT jobs:  {run_dir / 'run_pact.sh'}")


if __name__ == "__main__":
    main()
