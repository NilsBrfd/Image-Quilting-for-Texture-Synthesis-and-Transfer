from PIL import Image
import os
import numpy as np
from random import randint
import argparse


# ===================== ARGUMENTS =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Image quilting with minimum error boundary cut")

    parser.add_argument("--src", type=str, required=True, help="Texture image path")
    parser.add_argument("--patch", type=int, default=10, help="Number of patches per dimension")
    parser.add_argument("--overlap", type=float, default=30, help="Overlap percentage")
    parser.add_argument("--multiply", type=float, default=2, help="Output scaling factor")
    parser.add_argument("--samples", type=int, default=300, help="Number of random candidate patches")
    parser.add_argument("--cut", action="store_true", help="Display seam cut")
    return parser.parse_args()


args = parse_args()

IMAGE_SRC = args.src
NOMBRE_CARRE = args.patch
OVERLAP_PERCENT = args.overlap
MULTIPLY = args.multiply
NOMBRE_SAMPLES = args.samples
AFFICHE_CUT = args.cut


# ===================== CHECK FILE =====================

if not os.path.exists(IMAGE_SRC):
    raise FileNotFoundError(f"Texture image not found: {IMAGE_SRC}")


# ===================== ERRORS =====================

def get_mat_erreur(arr1, arr2):
    assert arr1.shape == arr2.shape, f"{arr1.shape} != {arr2.shape}"

    arr1f = arr1.astype(np.int32)
    arr2f = arr2.astype(np.int32)
    return (arr1f - arr2f) ** 2


def get_error_value(arr1, arr2):
    return np.sum(get_mat_erreur(arr1, arr2))


# ===================== VERTICAL SEAM =====================

def get_mat_erreur_chemin_vertical(mat_error):
    if mat_error.ndim == 3:
        mat_error = mat_error.mean(axis=2).astype(np.int32)
    else:
        mat_error = mat_error.astype(np.int32)

    H, W = mat_error.shape

    for i in range(1, H):
        for j in range(W):
            left = mat_error[i - 1, j - 1] if j - 1 >= 0 else np.inf
            center = mat_error[i - 1, j]
            right = mat_error[i - 1, j + 1] if j + 1 < W else np.inf
            mat_error[i, j] += min(left, center, right)

    return mat_error


def get_chemin_vertical(mat_error):
    mat_error = get_mat_erreur_chemin_vertical(mat_error)
    H, W = mat_error.shape
    chemin = np.zeros(H, dtype=int)
    chemin[-1] = np.argmin(mat_error[-1])

    for i in range(H - 2, -1, -1):
        j = chemin[i + 1]
        j0 = max(j - 1, 0)
        j1 = min(j + 2, W)
        chemin[i] = j0 + np.argmin(mat_error[i, j0:j1])

    return chemin


def get_overlap_cutted_vertical(arr1, arr2):
    mat_error = get_mat_erreur(arr1, arr2)
    chemin = get_chemin_vertical(mat_error)
    res = arr1.copy()
    H, W, _ = arr1.shape

    for i in range(H):
        for j in range(W):
            if j == chemin[i] and AFFICHE_CUT:
                res[i, j] = (0, 255, 0)
            elif j > chemin[i]:
                res[i, j] = arr2[i, j]

    return res


# ===================== HORIZONTAL SEAM =====================

def get_mat_erreur_chemin_horizontal(mat_error):
    if mat_error.ndim == 3:
        mat_error = mat_error.mean(axis=2).astype(np.int32)
    else:
        mat_error = mat_error.astype(np.int32)

    H, W = mat_error.shape

    for j in range(1, W):
        for i in range(H):
            up = mat_error[i - 1, j - 1] if i - 1 >= 0 else np.inf
            mid = mat_error[i, j - 1]
            down = mat_error[i + 1, j - 1] if i + 1 < H else np.inf
            mat_error[i, j] += min(up, mid, down)

    return mat_error


def get_chemin_horizontal(mat_error):
    mat_error = get_mat_erreur_chemin_horizontal(mat_error)
    H, W = mat_error.shape
    chemin = np.zeros(W, dtype=int)
    chemin[-1] = np.argmin(mat_error[:, -1])

    for j in range(W - 2, -1, -1):
        i = chemin[j + 1]
        i0 = max(i - 1, 0)
        i1 = min(i + 2, H)
        chemin[j] = i0 + np.argmin(mat_error[i0:i1, j])

    return chemin


def get_overlap_cutted_horizontal(arr1, arr2):
    mat_error = get_mat_erreur(arr1, arr2)
    chemin = get_chemin_horizontal(mat_error)
    res = arr1.copy()
    H, W, _ = arr1.shape

    for j in range(W):
        for i in range(H):
            if i == chemin[j] and AFFICHE_CUT:
                res[i, j] = (255, 0, 0)
            elif i > chemin[j]:
                res[i, j] = arr2[i, j]

    return res


# ===================== QUILTING =====================

def get_quilting(array, output, size_h, size_w, overlaps):
    overlap_h = max(2, int(size_h * overlaps))
    overlap_w = max(2, int(size_w * overlaps))
    step_h = size_h - overlap_h
    step_w = size_w - overlap_w

    for i in range(0, output.shape[0], step_h):
        for j in range(0, output.shape[1], step_w):
            patch_h = min(size_h, output.shape[0] - i)
            patch_w = min(size_w, output.shape[1] - j)
            current_overlap_h = min(overlap_h, patch_h)
            current_overlap_w = min(overlap_w, patch_w)
            best_patch = None
            best_error = float("inf")

            # ===================== FIRST PATCH =====================

            if i == 0 and j == 0:
                x = randint(0, array.shape[1] - patch_w)
                y = randint(0, array.shape[0] - patch_h)
                best_patch = array[y:y + patch_h, x:x + patch_w].copy()

            # ===================== FIRST ROW =====================

            elif i == 0:
                ref = output[i:i + patch_h, j:j + current_overlap_w]

                for _ in range(NOMBRE_SAMPLES):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)
                    candidate = array[y:y + patch_h, x:x + patch_w]
                    error = get_error_value(ref, candidate[:, :current_overlap_w])

                    if error < best_error:
                        best_error = error
                        best_patch = candidate.copy()

                best_patch[:, :current_overlap_w] = (
                    get_overlap_cutted_vertical(ref, best_patch[:, :current_overlap_w])
                )

            # ===================== FIRST COLUMN =====================

            elif j == 0:
                ref = output[i:i + current_overlap_h, j:j + patch_w]

                for _ in range(NOMBRE_SAMPLES):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)
                    candidate = array[y:y + patch_h, x:x + patch_w]
                    error = get_error_value(ref,candidate[:current_overlap_h, :])

                    if error < best_error:
                        best_error = error
                        best_patch = candidate.copy()

                best_patch[:current_overlap_h, :] = (get_overlap_cutted_horizontal(ref, best_patch[:current_overlap_h, :]))

            # ===================== CENTER PATCH =====================

            else:
                ref_w = output[
                    i:i + patch_h,
                    j:j + current_overlap_w
                ]

                ref_h = output[
                    i:i + current_overlap_h,
                    j:j + patch_w
                ]

                for _ in range(NOMBRE_SAMPLES):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)
                    candidate = array[y:y + patch_h, x:x + patch_w]
                    error = (get_error_value(ref_w, candidate[:, :current_overlap_w]) + get_error_value(ref_h, candidate[:current_overlap_h, :]))

                    if error < best_error:
                        best_error = error
                        best_patch = candidate.copy()

                best_patch[:, :current_overlap_w] = (get_overlap_cutted_vertical(ref_w, best_patch[:, :current_overlap_w]))
                best_patch[:current_overlap_h, :] = (get_overlap_cutted_horizontal(ref_h, best_patch[:current_overlap_h, :]))
            output[i:i + patch_h, j:j + patch_w] = best_patch

    return output


# ===================== MAIN =====================

im = Image.open(IMAGE_SRC).convert("RGB")
arr = np.array(im)

output_size = (round(arr.shape[0] * MULTIPLY), round(arr.shape[1] * MULTIPLY), 3)
empty = np.zeros(output_size, dtype=np.uint8)

block_h = output_size[0] // NOMBRE_CARRE
block_w = output_size[1] // NOMBRE_CARRE

print(f"Patch size: {block_h}x{block_w} | " f"Overlap: {OVERLAP_PERCENT}%")
result = get_quilting(arr,empty,block_h,block_w,OVERLAP_PERCENT / 100)
base_name = os.path.splitext(os.path.basename(IMAGE_SRC))[0]
save_dir = f"results/seam_cut_quilting/{base_name}_quilting"

os.makedirs(save_dir, exist_ok=True)
filename = (f"{base_name}_{NOMBRE_CARRE}_{OVERLAP_PERCENT}_{MULTIPLY}.png")
path = os.path.join(save_dir, filename)
Image.fromarray(result.astype(np.uint8)).save(path)
print("Saved:", path)
Image.fromarray(result.astype(np.uint8)).show()