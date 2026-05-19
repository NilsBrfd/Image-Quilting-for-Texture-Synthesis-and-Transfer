from PIL import Image
import os
import numpy as np
from random import randint
import argparse


# ===================== ARGUMENTS =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlap texture synthesis"
    )

    parser.add_argument("--src", type=str, required=True, help="Input texture")
    parser.add_argument("--patch", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=30)
    parser.add_argument("--multiply", type=float, default=2)

    return parser.parse_args()


args = parse_args()

IMAGE_SRC = args.src
NOMBRE_CARRE = args.patch
OVERLAP_PERCENT = args.overlap
MULTIPLY = args.multiply


# ===================== CHECK =====================

if not os.path.exists(IMAGE_SRC):
    raise FileNotFoundError(f"Image not found: {IMAGE_SRC}")


# ===================== ERROR =====================

def get_mat_erreur(arr1, arr2):
    assert arr1.shape == arr2.shape
    return (arr1.astype(np.int32) - arr2.astype(np.int32)) ** 2


def get_error_value(arr1, arr2):
    return np.sum(get_mat_erreur(arr1, arr2))


# ===================== OVERLAP QUILTING =====================

def get_overlap_quilting(array, output, size_h, size_w, overlap):

    overlap_h = max(2, int(size_h * overlap))
    overlap_w = max(2, int(size_w * overlap))

    step_h = size_h - overlap_h
    step_w = size_w - overlap_w

    H, W = output.shape[:2]

    for i in range(0, H, step_h):
        for j in range(0, W, step_w):

            patch_h = min(size_h, H - i)
            patch_w = min(size_w, W - j)

            ow = min(overlap_w, patch_w)
            oh = min(overlap_h, patch_h)

            best_patch = None
            best_error = float("inf")

            # ================= FIRST PATCH =================
            if i == 0 and j == 0:
                x = randint(0, array.shape[1] - patch_w)
                y = randint(0, array.shape[0] - patch_h)
                best_patch = array[y:y+patch_h, x:x+patch_w].copy()

            # ================= FIRST ROW =================
            elif i == 0:

                ref = output[i:i+patch_h, max(j-ow, 0):j]

                for _ in range(400):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)

                    cand = array[y:y+patch_h, x:x+patch_w]

                    error = np.sum(
                        (ref.astype(np.int32) - cand[:, :ref.shape[1]].astype(np.int32)) ** 2
                    )

                    if error < best_error:
                        best_error = error
                        best_patch = cand.copy()

            # ================= FIRST COLUMN =================
            elif j == 0:

                ref = output[max(i-oh, 0):i, j:j+patch_w]

                for _ in range(400):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)

                    cand = array[y:y+patch_h, x:x+patch_w]

                    error = np.sum(
                        (ref.astype(np.int32) - cand[:ref.shape[0], :].astype(np.int32)) ** 2
                    )

                    if error < best_error:
                        best_error = error
                        best_patch = cand.copy()

            # ================= CENTER =================
            else:

                ref_w = output[i:i+patch_h, j-ow:j]
                ref_h = output[i-oh:i, j:j+patch_w]

                for _ in range(400):
                    x = randint(0, array.shape[1] - patch_w)
                    y = randint(0, array.shape[0] - patch_h)

                    cand = array[y:y+patch_h, x:x+patch_w]

                    error = (
                        np.sum((ref_w.astype(np.int32) - cand[:, :ow].astype(np.int32)) ** 2) +
                        np.sum((ref_h.astype(np.int32) - cand[:oh, :].astype(np.int32)) ** 2)
                    )

                    if error < best_error:
                        best_error = error
                        best_patch = cand.copy()

            # ================= PLACE =================
            output[i:i+patch_h, j:j+patch_w] = best_patch

    return output


# ===================== MAIN =====================

im = Image.open(IMAGE_SRC).convert("RGB")
arr = np.array(im)

output_size = (round(arr.shape[0] * MULTIPLY), round(arr.shape[1] * MULTIPLY), 3)

output = np.zeros(output_size, dtype=np.uint8)

block_h = output_size[0] // NOMBRE_CARRE
block_w = output_size[1] // NOMBRE_CARRE

print(f"Output: {output_size}")
print(f"Patch: {block_h}x{block_w}")

result = get_overlap_quilting(arr,output,block_h,block_w,OVERLAP_PERCENT / 100)

base = os.path.splitext(os.path.basename(IMAGE_SRC))[0]

save_dir = f"results/overlap_quilting/{base}_overlap"
os.makedirs(save_dir, exist_ok=True)

path = os.path.join(save_dir,f"{base}_{NOMBRE_CARRE}_{OVERLAP_PERCENT}_{MULTIPLY}.png")

Image.fromarray(result.astype(np.uint8)).save(path)

print("Saved:", path)
Image.fromarray(result.astype(np.uint8)).show()