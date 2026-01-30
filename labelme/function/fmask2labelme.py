import cv2
import numpy as np
import json
import os
import base64
import shutil
import matplotlib.pyplot as plt

def mask_to_labelme(image_path, mask_path, output_json_path, label_names):
    """
    將 mask 影像轉換為 Labelme 格式的 JSON 檔案。

    Args:
        image_path (str): 原始影像路徑。
        mask_path (str): 輸入的 mask 影像路徑。
        output_json_path (str): 輸出的 JSON 文件路徑。
        label_names (dict): 標籤名稱與對應灰度值的字典。
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    shapes = []

    for value, label in label_names.items():
        binary_mask = np.uint8(mask == value)
        # contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(binary_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
        # new_contours = contours[1:]  # 從索引 1 開始切片
        for contour in contours:
            points = contour.squeeze().tolist()
            if len(points) < 3:
                continue

            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            })

    labelme_json = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_base64,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }

    with open(output_json_path, 'w') as f:
        json.dump(labelme_json, f, indent=4)


def process_dataset(dataset_dir, output_dir, label_names):
    """
    處理整個數據集，將所有 mask 影像轉換為 Labelme 格式的 JSON 文件，並複製對應的圖檔。

    Args:
        dataset_dir (str): 數據集根目錄。
        output_dir (str): 輸出目錄。
        label_names (dict): 標籤名稱與對應灰度值的字典。
    """
    masks_dir = os.path.join(dataset_dir, 'masks')
    images_dir = os.path.join(dataset_dir, 'images')

    for split in ['train', 'val']:
        split_masks_dir = os.path.join(masks_dir, split)
        split_images_dir = os.path.join(images_dir, split)

        for mask_filename in os.listdir(split_masks_dir):
            if mask_filename.endswith('.png'):
                mask_path = os.path.join(split_masks_dir, mask_filename)
                image_path = os.path.join(split_images_dir, mask_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Corresponding image for mask {mask_filename} not found.")
                    continue

                output_json_name = f"{os.path.splitext(mask_filename)[0]}_{split}.json"
                output_image_name = f"{os.path.splitext(mask_filename)[0]}_{split}.png"
                output_json_path = os.path.join(output_dir, output_json_name)
                output_image_path = os.path.join(output_dir, output_image_name)

                try:
                    # 轉換 mask 為 Labelme JSON
                    mask_to_labelme(image_path, mask_path, output_json_path, label_names)
                    # 複製原始圖像到輸出目錄
                    shutil.copy(image_path, output_image_path)
                    print(f"Converted {mask_filename} to {output_json_name} and copied {output_image_name}.")
                except Exception as e:
                    print(f"跳過 {mask_filename}：{e}")

def mask2labelme(dataset_dir, output_dir, label_names):
    os.makedirs(output_dir, exist_ok=True)
    process_dataset(dataset_dir, output_dir, label_names)
