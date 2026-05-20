import sys
import os
print(sys.path)
prj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, prj_path)
import cv2
import torch
import numpy as np
import onnxruntime as ort
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

# =========================================
# 1. Load ONNX model
# =========================================
session = ort.InferenceSession(r"F:\Tu_workspace\SGLATrack\onnx_models\sgla_op11_v5.onnx", 
                                providers=["CUDAExecutionProvider"])

preprocessor = Preprocessor()

# =========================================
# 2. Đọc video + chọn bbox ban đầu
# =========================================
cap = cv2.VideoCapture(r"F:\My_workspace\SGLATrack_copy\tracking\UAV-Anti-UAV_Train_000001\UAV-Anti-UAV_Train_000001.mp4")
ret, first_frame = cap.read()
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# Chọn bbox ban đầu (x, y, w, h)
# Chọn bbox ban đầu (x, y, w, h)
init_bbox = cv2.selectROI("Select Object", first_frame)
cv2.destroyAllWindows()
init_bbox = [float(v) for v in init_bbox]  # ← list of float

# =========================================
# 3. Initialize — xử lý template (1 lần)
# =========================================
template_factor = 2.0    # lấy từ params
template_size   = 128    # lấy từ cfg.TEST.TEMPLATE_SIZE

z_patch_arr, resize_factor_z, z_amask_arr = sample_target(
    first_frame_rgb, init_bbox, template_factor, output_sz=template_size
)
template_tensor = preprocessor.process(z_patch_arr, z_amask_arr)
template_np = template_tensor.tensors.detach().cpu().numpy()  # (1, 3, 128, 128)

state = init_bbox.copy()  # ← list.copy() thay vì list(tensor)

# =========================================
# 4. Track — lặp từng frame
# =========================================
search_factor = 4.0     # lấy từ params
search_size   = 256     # lấy từ cfg.TEST.SEARCH_SIZE

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crop search region
    x_patch_arr, resize_factor, x_amask_arr = sample_target(
        frame_rgb, state, search_factor, output_sz=search_size
    )
    search_tensor = preprocessor.process(x_patch_arr, x_amask_arr)
    #search_np = search_tensor.tensors.numpy()  # (1, 3, 256, 256)
    search_np = search_tensor.tensors.detach().cpu().numpy()

    # ONNX inference
    pred_boxes = session.run(
        ["pred_boxes"],
        {"template": template_np, "search": search_np}
    )

    # Tính bbox cuối — theo logic trong track()
    pred_boxes = torch.from_numpy(pred_boxes[0]).view(-1, 4)
    pred_box = (pred_boxes.mean(dim=0) * search_size / resize_factor).tolist()

    # Map về tọa độ ảnh gốc
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    state = clip_box(
        [cx_real - 0.5*w, cy_real - 0.5*h, w, h], H, W, margin=10
    )

    # Vẽ bbox
    x1, y1, w, h = [int(v) for v in state]
    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()