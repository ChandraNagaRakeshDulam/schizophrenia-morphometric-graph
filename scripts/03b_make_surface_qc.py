from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def ras_to_voxel(vertices, vox2ras_tkr):
    """Convert FreeSurfer surface tkrRAS coordinates to voxel coordinates."""
    inv = np.linalg.inv(vox2ras_tkr)
    xyz1 = np.c_[vertices, np.ones(len(vertices))]
    return (inv @ xyz1.T).T[:, :3]


def surface_points(subject_dir, hemi, surface, vox2ras_tkr):
    vertices, _ = nib.freesurfer.read_geometry(
        str(subject_dir / "surf" / f"{hemi}.{surface}")
    )
    return ras_to_voxel(vertices, vox2ras_tkr)


def plot_orientation(
    axes,
    volume,
    white,
    pial,
    orientation,
    slice_indices,
):
    for ax, idx in zip(axes, slice_indices):

        if orientation == "sagittal":
            img = volume[idx, :, :].T
            w = white[np.abs(white[:, 0] - idx) < 0.7]
            p = pial[np.abs(pial[:, 0] - idx) < 0.7]

            ax.scatter(w[:, 1], w[:, 2], s=0.4, c="yellow")
            ax.scatter(p[:, 1], p[:, 2], s=0.4, c="red")

        elif orientation == "coronal":
            img = volume[:, idx, :].T
            w = white[np.abs(white[:, 1] - idx) < 0.7]
            p = pial[np.abs(pial[:, 1] - idx) < 0.7]

            ax.scatter(w[:, 0], w[:, 2], s=0.4, c="yellow")
            ax.scatter(p[:, 0], p[:, 2], s=0.4, c="red")

        else:
            img = volume[:, :, idx].T
            w = white[np.abs(white[:, 2] - idx) < 0.7]
            p = pial[np.abs(pial[:, 2] - idx) < 0.7]

            ax.scatter(w[:, 0], w[:, 1], s=0.4, c="yellow")
            ax.scatter(p[:, 0], p[:, 1], s=0.4, c="red")

        ax.imshow(
            img,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
        )

        ax.set_title(f"{orientation} {idx}", fontsize=8)
        ax.axis("off")


def make_qc(subject_dir, out_file):

    brain_file = subject_dir / "mri" / "brainmask.mgz"

    if not brain_file.exists():
        raise FileNotFoundError(brain_file)

    img = nib.load(str(brain_file))
    volume = np.asarray(img.dataobj)

    vox2ras_tkr = img.header.get_vox2ras_tkr()

    white_parts = []
    pial_parts = []

    for hemi in ["lh", "rh"]:
        white_parts.append(
            surface_points(subject_dir, hemi, "white", vox2ras_tkr)
        )
        pial_parts.append(
            surface_points(subject_dir, hemi, "pial", vox2ras_tkr)
        )

    white = np.vstack(white_parts)
    pial = np.vstack(pial_parts)

    shape = volume.shape

    sagittal = np.linspace(
        int(shape[0] * 0.30),
        int(shape[0] * 0.70),
        5,
        dtype=int,
    )

    coronal = np.linspace(
        int(shape[1] * 0.25),
        int(shape[1] * 0.75),
        5,
        dtype=int,
    )

    axial = np.linspace(
        int(shape[2] * 0.30),
        int(shape[2] * 0.70),
        5,
        dtype=int,
    )

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    plot_orientation(
        axes[0], volume, white, pial, "sagittal", sagittal
    )

    plot_orientation(
        axes[1], volume, white, pial, "coronal", coronal
    )

    plot_orientation(
        axes[2], volume, white, pial, "axial", axial
    )

    fig.suptitle(
        f"{subject_dir.name}\n"
        "Yellow = white surface | Red = pial surface",
        fontsize=14,
    )

    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()

    print("Created:", out_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    subjects_dir = Path(args.subjects_dir)
    subject_dir = subjects_dir / args.subject
    out_dir = Path(args.out_dir)

    make_qc(
        subject_dir,
        out_dir / f"{args.subject}_surface_qc.png",
    )


if __name__ == "__main__":
    main()
