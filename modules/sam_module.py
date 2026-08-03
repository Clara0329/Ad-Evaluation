import cv2
import numpy as np

from ultralytics import YOLO
from segment_anything import sam_model_registry
from segment_anything import SamPredictor


# =========================
# 1. 加载YOLO
# =========================

yolo_model = YOLO("models/yolov8n.pt")


# =========================
# 2. 加载SAM
# =========================

sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
)

predictor = SamPredictor(sam)


# =========================
# 3. SAM分析函数
# =========================

def analyze_visual_focus(image_path):

    # 读取图片
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]

    # YOLO检测
    results = yolo_model(image)

    predictor.set_image(image_rgb)

    object_infos = []

    total_mask_area = 0

    # 遍历检测结果
    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        classes = result.boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, classes):

            # SAM分割
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=False
            )

            mask = masks[0]

            # 面积
            mask_area = np.sum(mask)

            total_mask_area += mask_area

            # 中心点
            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 是否在中心区域
            is_centered = (
                width * 0.3 < center_x < width * 0.7
                and
                height * 0.3 < center_y < height * 0.7
            )

            class_name = yolo_model.names[int(cls_id)]

            object_infos.append({
                "object": class_name,
                "area": int(mask_area),
                "is_centered": is_centered
            })

    # 主体面积占比
    image_area = width * height

    focus_ratio = total_mask_area / image_area

    result = {
        "object_count": len(object_infos),
        "focus_ratio": round(focus_ratio, 3),
        "objects": object_infos
    }

    return result