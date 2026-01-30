import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

# 圖像副檔名
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".JPG")


def lme2mask(input_folder, output_folder, log_callback=None):
    """log_callback(msg): 可選，用於將訊息（含出錯那一組的錯誤訊息）輸出到 UI 等處。"""
    def out(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    if not os.path.exists(input_folder):
        out("No data source. / 沒有數據來源。")
        sys.exit(0)

    out("All Files " + str(os.listdir(input_folder)))

    os.makedirs(output_folder + "masks/", exist_ok=True)
    os.makedirs(output_folder + "images/", exist_ok=True)

    key_to_json = {}
    key_to_image = {}
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            key = os.path.splitext(filename)[0]
            key_to_json[key] = filename
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(IMAGE_EXTS):
            key = os.path.splitext(filename)[0]
            key_to_image[key] = filename

    keys = sorted(set(key_to_json) & set(key_to_image))
    if not keys:
        out("沒有找到成對的 JSON 與圖像檔案。")
        return

    out(f"找到 {len(keys)} 組可處理的 JSON+圖像。")

    success_count = 0
    for key in keys:
        json_filename = key_to_json[key]
        image_filename = key_to_image[key]
        json_path = os.path.join(input_folder, json_filename)
        image_path = os.path.join(input_folder, image_filename)
        try:
            # 使用 PIL 讀取圖像（可正確處理檔名含 []、() 等字元的路徑；cv2.imread 在 Windows 上會失敗）
            pil_img = Image.open(image_path)
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask = np.zeros_like(image, dtype=np.uint8)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            points_save = []
            for i in range(len(data["shapes"])):
                points = np.array(data["shapes"][i]["points"], dtype=np.int32)
                points_save.append(points)
            for i in range(len(data["shapes"])):
                cv2.fillPoly(mask, [points_save[i]], (255, 255, 255))
            mask_out = os.path.join(output_folder + "masks/", key + ".png")
            cv2.imwrite(mask_out, mask)
            img_out = os.path.join(output_folder + "images/", key + ".png")
            pil_img.save(img_out, "PNG")
            success_count += 1
            out("INFO. 已處理: " + key)
        except Exception as e:
            out(f"✗ 跳過 {key}：{e}")

    out(f"完成：成功 {success_count} 組，共 {len(keys)} 組。")
