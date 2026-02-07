# import os
# import glob
# import cv2
# import numpy as np
# from PIL import Image
# import re

# # -----------------------------
# # Config
# # -----------------------------



# TEST_MASK_ROOT = (
#     "/export/livia/home/vision/Aguichemerre/datasets/"
#     "CAMELYON17_512/camelyon17/w-512xh-512/"
#     "metastatic-patches/testing/tumor"
# )

# def parse_cam_folder_name(folder_name):
#     """
#     Extract patient_node and patch filename from CAM folder name
#     """
#     m = re.search(r"(patient_\d+_node_\d+)", folder_name)
#     if m is None:
#         raise ValueError(f"Cannot parse patient/node from {folder_name}")
#     patient_node = m.group(1)

#     m2 = re.search(r"(file_testing-[^_]+_reg_.*)", folder_name)
#     if m2 is None:
#         raise ValueError(f"Cannot parse patch file from {folder_name}")
#     patch_file = m2.group(1)

#     return patient_node, patch_file





# root_dir = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON17_512/resnet50/STD_CL/PixelCAM/id_unlearning_c17_3_mask_w_aug_pxcam-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/tracked_cams/train"

# #root_dir = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON17_512/resnet50/STD_CL/PixelCAM/id_unlearning_c17_80-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/tracked_cams"


# out_path = "cams_aug_pxcam_train_unlearning.mp4"
# fps = 3

# N_PATCHES = 8
# GRID_ROWS = 2
# GRID_COLS = 4
# PAD = 20

# assert GRID_ROWS * GRID_COLS == N_PATCHES

# # -----------------------------
# # 1. Collect patch folders
# # -----------------------------
# patch_dirs = sorted([
#     d for d in glob.glob(os.path.join(root_dir, "*"))
#     if os.path.isdir(d)
# ])

# assert len(patch_dirs) >= N_PATCHES, "No CAM folders found"

# patch_dirs = patch_dirs[:N_PATCHES]

# print("Selected patches:")
# for d in patch_dirs:
#     print(" ", os.path.basename(d))

# # -----------------------------
# # 2. Collect epochs (from first patch)
# # -----------------------------
# epoch_files = sorted([
#     f for f in glob.glob(os.path.join(patch_dirs[0], "epoch_*.png"))
#     if not f.endswith("_mask.png")
# ])

# assert len(epoch_files) > 0, "No epoch images found"

# epochs = [os.path.basename(f) for f in epoch_files]

# # -----------------------------
# # 3. Read first frame to get size
# # -----------------------------
# img0 = Image.open(os.path.join(patch_dirs[0], epochs[0])).convert("RGB")
# img0 = np.array(img0)
# h, w, _ = img0.shape

# CELL_H = h
# CELL_W = 2 * w  # CAM | MASK

# canvas_h = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * PAD
# canvas_w = GRID_COLS * CELL_W + (GRID_COLS - 1) * PAD

# # -----------------------------
# # 4. Video writer
# # -----------------------------
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# video = cv2.VideoWriter(out_path, fourcc, fps, (canvas_w, canvas_h))

# # -----------------------------
# # 5. Build video
# # -----------------------------
# for t, epoch_name in enumerate(epochs):

#     canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

#     for i, patch_dir in enumerate(patch_dirs):

#         cam_path = os.path.join(patch_dir, epoch_name)
#         if not os.path.exists(cam_path):
#             continue

#         # --- Load CAM ---
#         cam = Image.open(cam_path).convert("RGB")
#         cam = np.array(cam)

#         # --- Load mask if exists ---

#         mask_path = cam_path.replace(".png", "_mask.png")

#         if os.path.exists(mask_path):
#             mask = Image.open(mask_path).convert("L")
#             mask = np.array(mask)

#             mask_rgb = np.stack([mask] * 3, axis=-1)
#             cell_img = np.concatenate([cam, mask_rgb], axis=1)
#         else:
#             # No mask → CAM + black padding
#             pad = np.zeros((h, w, 3), dtype=np.uint8)
#             cell_img = np.concatenate([cam, pad], axis=1)

#         # --- Placement ---
#         r = i // GRID_COLS
#         c = i % GRID_COLS

#         y0 = r * (CELL_H + PAD)
#         y1 = y0 + CELL_H
#         x0 = c * (CELL_W + PAD)
#         x1 = x0 + CELL_W

#         canvas[y0:y1, x0:x1] = cell_img

#         # Patch label
#         cv2.putText(
#             canvas,
#             f"P{i}",
#             (x0 + 10, y0 + 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )

#         # CAM / MASK labels
#         cv2.putText(canvas, "CAM", (x0 + 10, y0 + h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(canvas, "MASK", (x0 + w + 10, y0 + h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     # Epoch label
#     cv2.putText(
#         canvas,
#         f"Epoch {t}",
#         (20, canvas_h - 20),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.2,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA,
#     )

#     video.write(canvas[:, :, ::-1])  # RGB → BGR

# video.release()
# print("Video saved:", out_path)


import os
import glob
import re
import cv2
import numpy as np
from PIL import Image

# ============================================================
# MODE
# ============================================================
ETEST = False   # False = train (mask dans tracked_cams)
                # True  = test  (mask depuis dataset brut)

# ============================================================
# CONFIG
# ============================================================

root_dir = (
    "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/"
    "CAMELYON17_512/resnet50/STD_CL/PixelCAM/"
    "id_unlearning_c17_val_accuracy-tsk_STD_CL-"
    "ds_CAMELYON17_512-fold_3-mag_None-runmode_search-"
    "mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-"
    "ecd_resnet50/tracked_cams/train"
)

out_path = (
    "cams_test_sfda_unlearning.mp4" if ETEST
    else "cams_train_sfda_unlearning.mp4"
)

fps = 3

N_PATCHES = 15
GRID_ROWS = 3
GRID_COLS = 5
PAD = 20

assert GRID_ROWS * GRID_COLS == N_PATCHES

# ============================================================
# TEST DATASET ROOT (ONLY USED IF ETEST=True)
# ============================================================

TEST_MASK_ROOT = (
    "/export/livia/home/vision/Aguichemerre/datasets/"
    "CAMELYON17_512/camelyon17/w-512xh-512/"
    "metastatic-patches/testing/tumor"
)

# ============================================================
# HELPERS
# ============================================================

def parse_cam_folder_name(folder_name):
    """
    Extract patient_node and patch filename from CAM folder name
    """
    m = re.search(r"(patient_\d+_node_\d+)", folder_name)
    if m is None:
        raise ValueError(f"Cannot parse patient/node from {folder_name}")
    patient_node = m.group(1)

    # file_testing-...tif_reg_...
    m2 = re.search(r"(file_testing-.*\.tif_reg_.*)", folder_name)
    if m2 is None:
        raise ValueError(f"Cannot parse patch file from {folder_name}")
    patch_file = m2.group(1)

    return patient_node, patch_file


# ============================================================
# 1. COLLECT PATCH FOLDERS
# ============================================================

patch_dirs = sorted([
    d for d in glob.glob(os.path.join(root_dir, "*"))
    if os.path.isdir(d)
])

assert len(patch_dirs) >= N_PATCHES, "No CAM folders found"

patch_dirs = patch_dirs[:N_PATCHES]

print(f"[INFO] Video mode = {'TEST' if ETEST else 'TRAIN'}")
print("[INFO] Selected patches:")
for d in patch_dirs:
    print(" ", os.path.basename(d))

# ============================================================
# 2. COLLECT EPOCHS
# ============================================================

epoch_files = sorted([
    f for f in glob.glob(os.path.join(patch_dirs[0], "epoch_*.png"))
    if not f.endswith("_mask.png")
])

assert len(epoch_files) > 0, "No epoch images found"

epochs = [os.path.basename(f) for f in epoch_files]

# ============================================================
# 3. READ FIRST FRAME (SIZE)
# ============================================================

img0 = Image.open(os.path.join(patch_dirs[0], epochs[0])).convert("RGB")
img0 = np.array(img0)
h, w, _ = img0.shape

CELL_H = h
CELL_W = 2 * w  # CAM | MASK

canvas_h = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * PAD
canvas_w = GRID_COLS * CELL_W + (GRID_COLS - 1) * PAD

# ============================================================
# 4. VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(out_path, fourcc, fps, (canvas_w, canvas_h))

# ============================================================
# 5. BUILD VIDEO
# ============================================================

for t, epoch_name in enumerate(epochs):

    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for i, patch_dir in enumerate(patch_dirs):

        cam_path = os.path.join(patch_dir, epoch_name)
        if not os.path.exists(cam_path):
            continue

        # --------------------------
        # Load CAM
        # --------------------------
        cam = Image.open(cam_path).convert("RGB")
        cam = np.array(cam)

        # --------------------------
        # Resolve mask path
        # --------------------------
        mask_path = None

        if not ETEST:
            # -------- TRAIN MODE --------
            mask_path = cam_path.replace(".png", "_mask.png")
            complete_mask_path = mask_path

        else:
            # -------- TEST MODE --------
            folder_name = os.path.basename(patch_dir)
            try:
                patient_node, patch_file = parse_cam_folder_name(folder_name)
                mask_path = os.path.join(
                    TEST_MASK_ROOT,
                    patient_node,
                    f"mask_{patch_file}"
                )
                complete_mask_path = mask_path + ".png" if mask_path is not None else None
            except Exception as e:
                print(f"[WARN] {e}")
                mask_path = None

        # --------------------------
        # Load mask if exists
        # --------------------------
        
        if complete_mask_path is not None and os.path.exists(complete_mask_path):
            mask = Image.open(complete_mask_path).convert("L")
            # IMPORTANT: resize GT mask to CAM resolution
            mask = mask.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask)

            mask_rgb = np.stack([mask] * 3, axis=-1)
            cell_img = np.concatenate([cam, mask_rgb], axis=1)
        else:
            pad = np.zeros((h, w, 3), dtype=np.uint8)
            cell_img = np.concatenate([cam, pad], axis=1)

        # --------------------------
        # Placement
        # --------------------------
        r = i // GRID_COLS
        c = i % GRID_COLS

        y0 = r * (CELL_H + PAD)
        y1 = y0 + CELL_H
        x0 = c * (CELL_W + PAD)
        x1 = x0 + CELL_W

        canvas[y0:y1, x0:x1] = cell_img

        # Patch label
        cv2.putText(
            canvas,
            f"P{i}",
            (x0 + 10, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # CAM / MASK labels
        cv2.putText(
            canvas, "CAM",
            (x0 + 10, y0 + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
        cv2.putText(
            canvas, "MASK",
            (x0 + w + 10, y0 + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

    # Epoch label
    cv2.putText(
        canvas,
        f"Epoch {t}",
        (20, canvas_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    video.write(canvas[:, :, ::-1])  # RGB → BGR

video.release()
print("[INFO] Video saved:", out_path)


