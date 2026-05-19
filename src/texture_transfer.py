from PIL import Image
import os
import numpy as np
import cv2
from random import randint
import argparse


# ===================== ARGUMENTS =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Texture transfer Efros & Freeman")

    parser.add_argument("--src", type=str, required=True, help="Image source (target)")
    parser.add_argument("--dest", type=str, required=True, help="Image texture")

    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--patch", type=int, default=15)
    parser.add_argument("--overlap", type=float, default=30)
    parser.add_argument("--multiply", type=float, default=1)
    parser.add_argument("--cut", action="store_true")

    return parser.parse_args()


args = parse_args()

IMAGE_SRC = args.src
IMAGE_DEST = args.dest

NOMBRE_OCCURENCE = args.iter
NOMBRE_CARRE = args.patch
OVERLAP_PERCENT = args.overlap
MULTIPLY = args.multiply
AFFICHE_CUT = args.cut


# ===================== CHECK FILES =====================

if not os.path.exists(IMAGE_SRC):
    raise FileNotFoundError(f"Image source introuvable: {IMAGE_SRC}")

if not os.path.exists(IMAGE_DEST):
    raise FileNotFoundError(f"Image texture introuvable: {IMAGE_DEST}")


# ===================== FEATURE MAP =====================

def feature_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


# ===================== ERREURS =====================

def normalized_error(a, b):
    return np.sum((a.astype(np.int32) - b.astype(np.int32))**2) / (a.shape[0] * a.shape[1])


# ===================== SEAM CUT =====================

def get_mat_erreur(arr1, arr2):
    arr1f = arr1.astype(np.int32)
    arr2f = arr2.astype(np.int32)
    return (arr1f - arr2f)**2


def get_mat_erreur_chemin_vertical(mat_error):
    if mat_error.ndim == 3:
        mat_error = mat_error.mean(axis=2)

    H, W = mat_error.shape

    for i in range(1, H):
        for j in range(W):
            left   = mat_error[i-1, j-1] if j-1 >= 0 else np.inf
            center = mat_error[i-1, j]
            right  = mat_error[i-1, j+1] if j+1 < W else np.inf
            mat_error[i, j] += min(left, center, right)

    return mat_error


def get_chemin_vertical(mat_error):
    mat_error = get_mat_erreur_chemin_vertical(mat_error)
    H, W = mat_error.shape

    chemin = np.zeros(H, dtype=int)
    chemin[-1] = np.argmin(mat_error[-1])

    for i in range(H-2, -1, -1):
        j = chemin[i+1]
        j0 = max(j-1, 0)
        j1 = min(j+2, W)
        chemin[i] = j0 + np.argmin(mat_error[i, j0:j1])

    return chemin


def get_overlap_cutted_vertical(arr1, arr2):
    mat_error = get_mat_erreur(arr1, arr2)
    chemin = get_chemin_vertical(mat_error)

    res = arr1.copy()
    H, W, _ = arr1.shape

    for i in range(H):
        for j in range(W):
            if j > chemin[i]:
                res[i, j] = arr2[i, j]

    return res


def get_mat_erreur_chemin_horizontal(mat_error):
    if mat_error.ndim == 3:
        mat_error = mat_error.mean(axis=2)

    H, W = mat_error.shape

    for j in range(1, W):
        for i in range(H):
            up   = mat_error[i-1, j-1] if i-1 >= 0 else np.inf
            mid  = mat_error[i, j-1]
            down = mat_error[i+1, j-1] if i+1 < H else np.inf
            mat_error[i, j] += min(up, mid, down)

    return mat_error


def get_chemin_horizontal(mat_error):
    mat_error = get_mat_erreur_chemin_horizontal(mat_error)
    H, W = mat_error.shape

    chemin = np.zeros(W, dtype=int)
    chemin[-1] = np.argmin(mat_error[:, -1])

    for j in range(W-2, -1, -1):
        i = chemin[j+1]
        i0 = max(i-1, 0)
        i1 = min(i+2, H)
        chemin[j] = i0 + np.argmin(mat_error[i0:i1, j])

    return chemin


def get_overlap_cutted_horizontal(arr1, arr2):
    mat_error = get_mat_erreur(arr1, arr2)
    chemin = get_chemin_horizontal(mat_error)

    res = arr1.copy()
    H, W, _ = arr1.shape

    for j in range(W):
        for i in range(H):
            if i > chemin[j]:
                res[i, j] = arr2[i, j]

    return res


# ===================== TRANSFER =====================

def get_transfer(array_src, array_dest, size_h, size_w, overlaps, ALPHA):

    feat_src = feature_map(array_src)
    feat_dest = feature_map(array_dest)

    overlap_h = max(2, int(size_h * overlaps))
    overlap_w = max(2, int(size_w * overlaps))

    step_h = size_h - overlap_h
    step_w = size_w - overlap_w

    for i in range(0, array_src.shape[0], step_h):
        for j in range(0, array_src.shape[1], step_w):

            h = min(size_h, array_src.shape[0] - i)
            w = min(size_w, array_src.shape[1] - j)

            best_patch = None
            best_error = float("inf")

            ref_full = feat_src[i:i+h, j:j+w]

            for _ in range(200):
                y = randint(0, array_dest.shape[0] - h)
                x = randint(0, array_dest.shape[1] - w)

                cand = array_dest[y:y+h, x:x+w]
                cand_feat = feat_dest[y:y+h, x:x+w]

                error = normalized_error(ref_full, cand_feat)

                if i > 0:
                    error += ALPHA * normalized_error(
                        feat_src[i:i+overlap_h, j:j+w],
                        cand_feat[:overlap_h, :]
                    )

                if j > 0:
                    error += ALPHA * normalized_error(
                        feat_src[i:i+h, j:j+overlap_w],
                        cand_feat[:, :overlap_w]
                    )

                if error < best_error:
                    best_error = error
                    best_patch = cand.copy()

            if i > 0:
                best_patch[:overlap_h, :] = get_overlap_cutted_horizontal(
                    array_src[i:i+overlap_h, j:j+w],
                    best_patch[:overlap_h, :]
                )

            if j > 0:
                best_patch[:, :overlap_w] = get_overlap_cutted_vertical(
                    array_src[i:i+h, j:j+overlap_w],
                    best_patch[:, :overlap_w]
                )

            array_src[i:i+h, j:j+w] = best_patch

    return array_src


# ===================== MAIN =====================

im_src = Image.open(IMAGE_SRC).convert("RGB")
im_dest = Image.open(IMAGE_DEST).convert("RGB")

base_src = os.path.splitext(os.path.basename(IMAGE_SRC))[0]
base_dest = os.path.splitext(os.path.basename(IMAGE_DEST))[0]

test_previous = None

for i in range(NOMBRE_OCCURENCE):

    arr_dest = np.array(im_dest)

    current = np.array(im_src) if i == 0 else test_previous

    if NOMBRE_OCCURENCE > 1:
        ALPHA = 0.8 * i / (NOMBRE_OCCURENCE - 1) + 0.1
    else:
        ALPHA = 0.1

    base_h = current.shape[0] // NOMBRE_CARRE
    base_w = current.shape[1] // NOMBRE_CARRE

    scale = (2/3) ** i
    block_h = max(4, int(base_h * scale))
    block_w = max(4, int(base_w * scale))

    print(f"[Itération {i}] block={block_h}x{block_w}, alpha={ALPHA:.2f}")

    result = get_transfer(current.copy(), arr_dest, block_h, block_w, OVERLAP_PERCENT / 100, ALPHA)

    test_previous = result

    save_dir = f"results/texture_transfer/{base_src}_to_{base_dest}"
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"iter_{i}.png")
    Image.fromarray(result.astype(np.uint8)).save(path)

    print("Saved:", path)