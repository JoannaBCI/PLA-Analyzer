import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib

from skimage import filters, measure, segmentation, morphology, color
from skimage.io import imread
from scipy import ndimage as ndi


def load_image(path: str) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        img = img.max(axis=0)
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = color.rgb2gray(img[..., :3])
        img = (img * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)
    return img


def segment_nuclei(image: np.ndarray,
                   gaussian_sigma: float = 2.0,
                   min_nucleus_area_um2: float = 30.0,
                   pixel_size_um: float = 1.0) -> np.ndarray:
    blurred = filters.gaussian(image, sigma=gaussian_sigma)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = ndi.binary_fill_holes(binary)
    min_area_px = int(min_nucleus_area_um2 / (pixel_size_um ** 2))
    binary = morphology.remove_small_objects(binary, min_size=min_area_px)
    distance = ndi.distance_transform_edt(binary)
    coords = measure.peak_local_max(distance, footprint=np.ones((9, 9)), labels=binary)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labeled = segmentation.watershed(-distance, markers, mask=binary)
    return labeled


def segment_puncta(pla_image: np.ndarray,
                   gaussian_sigma: float = 1.0,
                   min_dot_area_um2: float = 0.3,
                   max_dot_area_um2: float = 10.0,
                   pixel_size_um: float = 1.0) -> np.ndarray:
    blurred = filters.gaussian(pla_image, sigma=gaussian_sigma)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    min_px = max(1, int(min_dot_area_um2 / (pixel_size_um ** 2)))
    max_px = int(max_dot_area_um2 / (pixel_size_um ** 2))
    labeled = measure.label(binary)
    for region in measure.regionprops(labeled):
        if region.area < min_px or region.area > max_px:
            labeled[labeled == region.label] = 0
    labeled = measure.label(labeled > 0)
    return labeled


def count_puncta_per_nucleus(labeled_nuclei: np.ndarray,
                              labeled_puncta: np.ndarray) -> dict:
    nucleus_ids = np.unique(labeled_nuclei)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    counts = {int(nid): 0 for nid in nucleus_ids}
    for dot_region in measure.regionprops(labeled_puncta):
        cy, cx = dot_region.centroid
        cy, cx = int(round(cy)), int(round(cx))
        if 0 <= cy < labeled_nuclei.shape[0] and 0 <= cx < labeled_nuclei.shape[1]:
            nucleus_at_dot = labeled_nuclei[cy, cx]
            if nucleus_at_dot != 0:
                counts[int(nucleus_at_dot)] += 1
    return counts


def make_overlay(dapi_image, pla_image, labeled_nuclei, labeled_puncta, counts, out_path):
    def norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img, dtype=float)

    dapi_norm = norm(dapi_image)
    pla_norm = norm(pla_image)

    rgb = np.stack([
        pla_norm * 0.6,
        dapi_norm * 0.0,
        dapi_norm * 0.7,
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb)

    boundaries = segmentation.find_boundaries(labeled_nuclei, mode="outer")
    boundary_overlay = np.zeros((*dapi_image.shape, 4), dtype=float)
    boundary_overlay[boundaries, :] = [1, 1, 1, 0.8]
    ax.imshow(boundary_overlay)

    if counts:
        max_count = max(counts.values()) if max(counts.values()) > 0 else 1
        cmap = matplotlib.colormaps["RdYlGn_r"]
        dot_overlay = np.zeros((*dapi_image.shape, 4), dtype=float)
        for dot_region in measure.regionprops(labeled_puncta):
            cy, cx = dot_region.centroid
            cy_i, cx_i = int(round(cy)), int(round(cx))
            if 0 <= cy_i < labeled_nuclei.shape[0] and 0 <= cx_i < labeled_nuclei.shape[1]:
                nuc = labeled_nuclei[cy_i, cx_i]
                dot_count = counts.get(int(nuc), 0)
                rgba = cmap(dot_count / max_count)
                coords = dot_region.coords
                dot_overlay[coords[:, 0], coords[:, 1]] = rgba
        ax.imshow(dot_overlay, alpha=0.9)

    for nid, cnt in counts.items():
        region_mask = labeled_nuclei == nid
        props = measure.regionprops(region_mask.astype(int))
        if props:
            cy, cx = props[0].centroid
            ax.text(cx, cy, f"#{nid}\n{cnt} dots",
                    color="white", fontsize=7, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.4, ec="none"))

    ax.set_title("PLA Overlay: DAPI (blue) + PLA dots (green=low, red=high)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Overlay saved: {out_path}")


def make_plot(counts, cell_type, out_path):
    if not counts:
        print("No nuclei found; skipping plot.")
        return

    values = list(counts.values())
    mean_val = np.mean(values)
    sem_val = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar([cell_type], [mean_val], yerr=[sem_val],
           color="#4C9BE8", alpha=0.75, capsize=6,
           width=0.4, error_kw={"elinewidth": 1.5, "ecolor": "black"})

    rng = np.random.default_rng(42)
    x_jitter = rng.uniform(-0.12, 0.12, size=len(values))
    ax.scatter(np.zeros(len(values)) + x_jitter, values,
               color="black", s=25, zorder=5, alpha=0.8)

    ax.set_ylabel("PLA dots / nucleus", fontsize=11)
    ax.set_title(f"PLA Quantification\nn = {len(values)} nuclei", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.text(0, mean_val + sem_val + (ax.get_ylim()[1] * 0.02),
            f"{mean_val:.2f} +/- {sem_val:.2f}",
            ha="center", va="bottom", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def run_pla_analysis(dapi_path, pla_path, cell_type="Sample",
                     pixel_size_um=1.0, output_dir=None):
    dapi_path = Path(dapi_path)
    pla_path = Path(pla_path)

    if output_dir is None:
        out_dir = dapi_path.parent / (dapi_path.stem + "_PLA_results")
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== PLA Analyzer ===")
    print(f"  DAPI : {dapi_path}")
    print(f"  PLA  : {pla_path}")
    print(f"  Output: {out_dir}\n")

    print("[1/5] Loading images...")
    dapi_img = load_image(str(dapi_path))
    pla_img = load_image(str(pla_path))

    print("[2/5] Segmenting nuclei (DAPI)...")
    labeled_nuclei = segment_nuclei(dapi_img, pixel_size_um=pixel_size_um)
    n_nuclei = labeled_nuclei.max()
    print(f"       Found {n_nuclei} nuclei.")

    print("[3/5] Segmenting PLA puncta...")
    labeled_puncta = segment_puncta(pla_img, pixel_size_um=pixel_size_um)
    n_dots = labeled_puncta.max()
    print(f"       Found {n_dots} PLA dots total.")

    print("[4/5] Counting dots per nucleus...")
    counts = count_puncta_per_nucleus(labeled_nuclei, labeled_puncta)
    mean_ratio = np.mean(list(counts.values())) if counts else 0
    print(f"       Mean dots/nucleus: {mean_ratio:.2f}")

    print("[5/5] Saving results...")
    make_overlay(dapi_img, pla_img, labeled_nuclei, labeled_puncta, counts,
                 out_path=str(out_dir / "pla_overlay.png"))
    make_plot(counts, cell_type=cell_type, out_path=str(out_dir / "pla_dots_per_nucleus.png"))

    df = pd.DataFrame([
        {"nucleus_id": nid, "dot_count": cnt, "cell_type": cell_type}
        for nid, cnt in counts.items()
    ])
    csv_path = out_dir / "pla_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    print(f"\nDone. {n_nuclei} nuclei | {n_dots} total dots | mean {mean_ratio:.2f} dots/nucleus")
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="PLA image analyzer: count fluorescent dots per nucleus."
    )
    p.add_argument("--dapi", required=True, help="Path to DAPI channel image")
    p.add_argument("--pla", required=True, help="Path to PLA channel image")
    p.add_argument("--cell-type", default="Sample", help="Label for plot x-axis")
    p.add_argument("--pixel-size", type=float, default=1.0, help="Microns per pixel")
    p.add_argument("--output-dir", default=None, help="Where to save results")
    return p.parse_args()


def main():
    args = parse_args()
    run_pla_analysis(
        dapi_path=args.dapi,
        pla_path=args.pla,
        cell_type=args.cell_type,
        pixel_size_um=args.pixel_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()