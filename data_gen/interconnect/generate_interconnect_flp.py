import os
import argparse
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def save_layout_png(df, out_png, title, die_size_um=2000.0, cu_label="Cu", diel_label="Diel"):
    """
    Save floorplan visualization as a colored layout PNG.
    This is optional and mainly for debugging geometry.
    Coordinates in dataframe are meters.
    Plot coordinates are micrometers.
    """
    if not HAS_MPL:
        print(f"Matplotlib not available. Skipping PNG: {out_png}")
        return

    color_map = {
        cu_label: "#d95f02",
        diel_label: "#d9d9d9",
    }

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    for _, r in df.iterrows():
        label = str(r["Label"])
        x_um = float(r["X"]) * 1e6
        y_um = float(r["Y"]) * 1e6
        w_um = float(r["Length (m)"]) * 1e6
        h_um = float(r["Width (m)"]) * 1e6

        facecolor = color_map.get(label, "#ffffff")
        edgecolor = "black" if label == cu_label else "white"
        linewidth = 0.15 if label == cu_label else 0.05

        rect = patches.Rectangle(
            (x_um, y_um),
            w_um,
            h_um,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, die_size_um)
    ax.set_ylim(0, die_size_um)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_title(title)

    cu_patch = patches.Patch(color=color_map[cu_label], label=cu_label)
    diel_patch = patches.Patch(color=color_map[diel_label], label=diel_label)
    ax.legend(handles=[cu_patch, diel_patch], loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved layout PNG: {out_png}")


def build_gray_raster_from_df(
    df,
    grid_n,
    die_size_um=2000.0,
    cu_label="Cu",
    diel_label="Diel",
    cu_value=2.56e-3,
    diel_value=0.724,
):
    """
    Convert the geometry dataframe to a grid_n x grid_n grayscale raster.

    400x400 means a 400 by 400 downsampled raster over the full 2mm x 2mm die.
    2000x2000 means a 2000 by 2000 raster over the same die.

    Cu pixels are assigned cu_value and Diel pixels are assigned diel_value.
    """
    cell_um = die_size_um / grid_n
    img = np.full((grid_n, grid_n), diel_value, dtype=np.float32)

    # Fill dielectric first, then overwrite Cu.
    for target_label, target_value in [(diel_label, diel_value), (cu_label, cu_value)]:
        sub = df[df["Label"] == target_label]
        for _, r in sub.iterrows():
            x0 = float(r["X"]) * 1e6
            y0 = float(r["Y"]) * 1e6
            w = float(r["Length (m)"]) * 1e6
            h = float(r["Width (m)"]) * 1e6

            x1 = x0 + w
            y1 = y0 + h

            ix0 = max(0, int(np.floor(x0 / cell_um)))
            iy0 = max(0, int(np.floor(y0 / cell_um)))
            ix1 = min(grid_n, int(np.ceil(x1 / cell_um)))
            iy1 = min(grid_n, int(np.ceil(y1 / cell_um)))

            img[iy0:iy1, ix0:ix1] = target_value

    return img


def save_gray_raster_png(
    df,
    out_png,
    grid_n,
    die_size_um=2000.0,
    cu_label="Cu",
    diel_label="Diel",
    cu_value=2.56e-3,
    diel_value=0.724,
):
    """
    Save grayscale raster PNG with no labels, no legend, no axes.
    """
    if not HAS_MPL:
        print(f"Matplotlib not available. Skipping raster PNG: {out_png}")
        return

    img = build_gray_raster_from_df(
        df=df,
        grid_n=grid_n,
        die_size_um=die_size_um,
        cu_label=cu_label,
        diel_label=diel_label,
        cu_value=cu_value,
        diel_value=diel_value,
    )

    plt.imsave(
        out_png,
        img,
        cmap="gray",
        vmin=min(cu_value, diel_value),
        vmax=max(cu_value, diel_value),
    )
    print(f"Saved raster PNG: {out_png}")


def sanity_check_and_save(df, out_csv, layer_name, die_size_um=2000.0, cu_label="Cu"):
    """
    Save CSV after checking:
      - bbox inside die
      - area covers die exactly
      - no negative coordinates
    This does not explicitly check pairwise overlaps, but construction guarantees non-overlap.
    """
    die_size_m = die_size_um * 1e-6
    die_area = die_size_m * die_size_m

    x_min = df["X"].min()
    y_min = df["Y"].min()
    x_max = (df["X"] + df["Length (m)"]).max()
    y_max = (df["Y"] + df["Width (m)"]).max()

    total_area = (df["Length (m)"] * df["Width (m)"]).sum()
    area_error = abs(total_area - die_area) / die_area

    cu_area = (
        df[df["Label"] == cu_label]["Length (m)"]
        * df[df["Label"] == cu_label]["Width (m)"]
    ).sum()
    cu_fill_ratio = cu_area / die_area

    if x_min < -1e-15 or y_min < -1e-15:
        raise ValueError(f"{layer_name}: negative coordinate detected.")
    if x_max > die_size_m + 1e-15 or y_max > die_size_m + 1e-15:
        raise ValueError(f"{layer_name}: floorplan exceeds die boundary.")
    if area_error > 1e-9:
        raise ValueError(
            f"{layer_name}: area coverage error={area_error:.3e}. "
            "There may be gaps or overlaps."
        )

    df.to_csv(out_csv, index=False, float_format="%.10f")

    print(f"Saved CSV: {out_csv}")
    print(f"  layer             : {layer_name}")
    print(f"  rectangles        : {len(df)}")
    print(f"  Cu fill ratio     : {cu_fill_ratio * 100:.3f}%")
    print(f"  bbox x            : {x_min:.6e} to {x_max:.6e}")
    print(f"  bbox y            : {y_min:.6e} to {y_max:.6e}")
    print(f"  area error        : {area_error:.3e}")
    print("")


def make_axis_segments(die_size_um, array_n, cu_size_um):
    """
    Create 1D non-overlapping segments along one axis.

    Each Cu interval is centered inside one pitch cell.
    The returned segments cover [0, die_size_um] exactly.
    """
    pitch_um = die_size_um / array_n

    if cu_size_um > pitch_um:
        raise ValueError(
            f"cu_size_um={cu_size_um} exceeds pitch={pitch_um}. "
            "Cu blocks would overlap."
        )

    cu_intervals = []
    for i in range(array_n):
        cell_start = i * pitch_um
        cu_start = cell_start + (pitch_um - cu_size_um) / 2.0
        cu_end = cu_start + cu_size_um
        cu_intervals.append((cu_start, cu_end))

    boundaries = [0.0, die_size_um]
    for a, b in cu_intervals:
        boundaries.append(a)
        boundaries.append(b)

    boundaries = sorted(boundaries)

    clean = []
    eps = 1e-9
    for x in boundaries:
        if not clean or abs(x - clean[-1]) > eps:
            clean.append(x)

    segments = []
    for a, b in zip(clean[:-1], clean[1:]):
        if b - a <= eps:
            continue

        mid = 0.5 * (a + b)
        is_cu = any((mid >= c0 - eps and mid <= c1 + eps) for c0, c1 in cu_intervals)

        segments.append({
            "start": a,
            "end": b,
            "is_cu": is_cu,
        })

    return segments, pitch_um


def generate_intersection_grid_flp(
    out_csv,
    layer_name,
    die_size_um=2000.0,
    array_n=20,
    cu_size_um=50.0,
    cu_label="Cu",
    diel_label="Diel",
    save_layout=False,
    out_layout_png=None,
):
    """
    Generate a non-overlapping grid floorplan.

    Rule:
      label = Cu only if x-segment is Cu and y-segment is Cu.
      otherwise label = Diel.
    """
    x_segments, pitch_um = make_axis_segments(die_size_um, array_n, cu_size_um)
    y_segments, _ = make_axis_segments(die_size_um, array_n, cu_size_um)

    rows = []
    cu_count = 0
    diel_count = 0

    for ys in y_segments:
        for xs in x_segments:
            x0 = xs["start"]
            x1 = xs["end"]
            y0 = ys["start"]
            y1 = ys["end"]

            w = x1 - x0
            h = y1 - y0

            if w <= 0 or h <= 0:
                continue

            if xs["is_cu"] and ys["is_cu"]:
                label = cu_label
                name = f"{layer_name}_CU_{cu_count:05d}"
                cu_count += 1
            else:
                label = diel_label
                name = f"{layer_name}_DIEL_{diel_count:05d}"
                diel_count += 1

            rows.append({
                "UnitName": name,
                "X": x0 * 1e-6,
                "Y": y0 * 1e-6,
                "Length (m)": w * 1e-6,
                "Width (m)": h * 1e-6,
                "ConfigFile": "",
                "Label": label,
            })

    df = pd.DataFrame(rows)

    print(f"Generating {layer_name} intersection grid:")
    print(f"  die size          : {die_size_um:.1f} um x {die_size_um:.1f} um")
    print(f"  array             : {array_n} x {array_n}")
    print(f"  pitch             : {pitch_um:.3f} um")
    print(f"  Cu size           : {cu_size_um:.3f} um")
    print(f"  Cu blocks         : {cu_count}")
    print(f"  Diel blocks       : {diel_count}")

    sanity_check_and_save(
        df=df,
        out_csv=out_csv,
        layer_name=layer_name,
        die_size_um=die_size_um,
        cu_label=cu_label,
    )

    if save_layout and out_layout_png:
        save_layout_png_fn = save_layout_png
        save_layout_png_fn(
            df=df,
            out_png=out_layout_png,
            title=f"{layer_name}: {array_n}x{array_n}, Cu size {cu_size_um:g} um",
            die_size_um=die_size_um,
            cu_label=cu_label,
            diel_label=diel_label,
        )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="interconnect_flp_grid")
    parser.add_argument("--die-size-um", type=float, default=2000.0)
    parser.add_argument("--cu-label", type=str, default="Cu")
    parser.add_argument("--diel-label", type=str, default="Diel")
    parser.add_argument("--cu-value", type=float, default=2.56e-3)
    parser.add_argument("--diel-value", type=float, default=0.724)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_png = not args.no_png

    # ============================================================
    # 1. Micro-bump (pitch 50um, diameter 30um)
    # ============================================================
    micro_csv = os.path.join(args.out_dir, "microbump_40x40_30um_flp.csv")
    micro_png_400 = os.path.join(args.out_dir, "microbump_400x400_gray.png")
    micro_png_2000 = os.path.join(args.out_dir, "microbump_2000x2000_gray.png")

    micro_df = generate_intersection_grid_flp(
        out_csv=micro_csv,
        layer_name="Microbump",
        die_size_um=args.die_size_um,
        array_n=40,          # 2000 / 50 = 40
        cu_size_um=30.0,
        cu_label=args.cu_label,
        diel_label=args.diel_label,
        save_layout=False,
        out_layout_png=None,
    )

    if save_png:
        save_gray_raster_png(
            df=micro_df,
            out_png=micro_png_400,
            grid_n=400,
            die_size_um=args.die_size_um,
            cu_label=args.cu_label,
            diel_label=args.diel_label,
            cu_value=args.cu_value,
            diel_value=args.diel_value,
        )
        save_gray_raster_png(
            df=micro_df,
            out_png=micro_png_2000,
            grid_n=2000,
            die_size_um=args.die_size_um,
            cu_label=args.cu_label,
            diel_label=args.diel_label,
            cu_value=args.cu_value,
            diel_value=args.diel_value,
        )

    # ============================================================
    # 2. Hybrid bonding (pitch 10um, diameter 5um)
    # ============================================================
    hybrid_csv = os.path.join(args.out_dir, "hybrid_100x100_5um_flp.csv")
    hybrid_png_400 = os.path.join(args.out_dir, "hybrid_400x400_gray.png")
    hybrid_png_2000 = os.path.join(args.out_dir, "hybrid_2000x2000_gray.png")

    hybrid_df = generate_intersection_grid_flp(
        out_csv=hybrid_csv,
        layer_name="Hybrid",
        die_size_um=args.die_size_um,
        array_n=200,         # 2000 / 10 = 200
        cu_size_um=5.0,
        cu_label=args.cu_label,
        diel_label=args.diel_label,
        save_layout=False,
        out_layout_png=None,
    )

    if save_png:
        save_gray_raster_png(
            df=hybrid_df,
            out_png=hybrid_png_400,
            grid_n=400,
            die_size_um=args.die_size_um,
            cu_label=args.cu_label,
            diel_label=args.diel_label,
            cu_value=args.cu_value,
            diel_value=args.diel_value,
        )
        save_gray_raster_png(
            df=hybrid_df,
            out_png=hybrid_png_2000,
            grid_n=2000,
            die_size_um=args.die_size_um,
            cu_label=args.cu_label,
            diel_label=args.diel_label,
            cu_value=args.cu_value,
            diel_value=args.diel_value,
        )


if __name__ == "__main__":
    main()
