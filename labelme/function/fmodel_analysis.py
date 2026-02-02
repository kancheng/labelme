"""
模型分析：與 segmentation-and-detection/yolo-segmentation.py 流程一致。
使用訓練好的 YOLO 分割模型對 Predict 目錄逐張推論（save_txt 產出 txt），
將 txt 轉為 mask 圖，與 Ground Truth 目錄計算每張 IOU、DSC，
結果輸出至 CSV 與 result.txt；可選分析輸出目錄保存預測 labels 與 masks。

IOU 用的「預測圖」如何生成：
1. 對 Predict 目錄每張圖做 model.predict(..., save_txt=True) → Ultralytics 寫出 txt 到 work_dir/predict/labels/（或 segment/predict/labels/）。
2. _yolo2maskdir_all(label_dir, predict_dir, mask_dir)：用「原圖目錄」predict_dir 的圖取得尺寸 (h,w)，用 label_dir 的 txt 讀取多邊形（歸一化座標），
   在與原圖同尺寸的黑底上以 cv2.fillPoly 畫白多邊形 → 存成灰階二值圖到 mask_dir（每張檔名與原圖主檔名相同、副檔名 .png）。
3. evaluate_miou_mdice_exclude_zero(mask_dir, gt_dir, ...)：比對 mask_dir（預測 mask 圖）與 gt_dir（Ground Truth mask 圖）同檔名的圖，算 IOU/DSC。
因此「IOU 的圖片」= 上述步驟 2 寫入 mask_dir 的預測 mask 圖；若未指定 output_dir 則用暫存目錄，分析結束後會刪除；有指定 output_dir 則寫入 output_dir/masks/ 可保留。
"""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# 二值化門檻（與評估一致）
THRESHOLD = 127
NUM_CLASSES = 1  # 二值分割


def calculate_iou_from_arrays(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """從二值化陣列 (0/1) 計算 IoU。"""
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def dice_coef(pred: np.ndarray, target: np.ndarray, num_classes: int = 1) -> float:
    """
    計算 Dice 係數。二值分割 (num_classes=1) 時只算前景 (class 1)，與 IoU 語義一致。
    """
    smooth = 1.0
    dice = 0.0
    if num_classes == 1:
        classes_to_use = [1]
    else:
        classes_to_use = list(range(num_classes))
    for cls in classes_to_use:
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice += (2.0 * intersection + smooth) / (union + smooth)
    return dice / len(classes_to_use)


def evaluate_miou_mdice_exclude_zero(
    predict_dir: str,
    ground_truth_dir: str,
    pred_ext: str = ".png",
    gt_ext: str = ".png",
    num_classes: int = NUM_CLASSES,
    min_iou: float = 0.5,
    min_dice: float = 0.5,
) -> Tuple[List[Tuple[str, float, float]], float, float]:
    """
    計算每張圖的 IoU 與 DSC；低於 min_iou 或 min_dice 的樣本排除於總 IOU、總 DSC。
    回傳：(per_image 全部列表, 總平均 IoU, 總平均 DSC)。
    """
    gt_files = sorted(
        f
        for f in os.listdir(ground_truth_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    per_image: List[Tuple[str, float, float]] = []

    for gt_file in gt_files:
        gt_path = os.path.join(ground_truth_dir, gt_file)
        base = os.path.splitext(gt_file)[0]
        pred_file = base + pred_ext
        pred_path = os.path.join(predict_dir, pred_file)

        if not os.path.exists(pred_path):
            continue

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None or gt_img is None:
            continue
        if pred_img.shape != gt_img.shape:
            continue

        _, pred_bin_255 = cv2.threshold(pred_img, THRESHOLD, 255, cv2.THRESH_BINARY)
        _, gt_bin_255 = cv2.threshold(gt_img, THRESHOLD, 255, cv2.THRESH_BINARY)
        pred_bin = pred_bin_255 // 255
        gt_bin = gt_bin_255 // 255

        iou = calculate_iou_from_arrays(pred_bin, gt_bin)
        dice = dice_coef(pred_bin, gt_bin, num_classes)
        per_image.append((os.path.basename(gt_file), iou, dice))

    # 排除低於門檻的樣本，僅對其餘取平均
    valid = [x for x in per_image if x[1] >= min_iou and x[2] >= min_dice]
    if valid:
        total_iou = float(np.mean([x[1] for x in valid]))
        total_dice = float(np.mean([x[2] for x in valid]))
    else:
        total_iou = 0.0
        total_dice = 0.0

    return per_image, total_iou, total_dice


def _read_txt_labels(txt_file: str) -> List[Tuple[int, List[float]]]:
    """讀取 YOLO 分割 txt 標註（class_id + 歸一化多邊形座標）。"""
    labels = []
    if not os.path.isfile(txt_file):
        return labels
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + 至少 3 點 (6 個數)
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            labels.append((class_id, coords))
    return labels


def _draw_labels(mask: np.ndarray, labels: List[Tuple[int, List[float]]], h: int, w: int) -> None:
    """將 YOLO 分割多邊形繪製到 mask 上（白 255）。"""
    for _cls, coordinates in labels:
        if len(coordinates) < 6:
            continue
        points = [
            (int(x * w), int(y * h))
            for x, y in zip(coordinates[::2], coordinates[1::2])
        ]
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))


def _yolo2mask_one(image_path: str, txt_path: str, out_path: str) -> None:
    """
    單張圖：依 YOLO txt 畫出 IOU 用的預測 mask 並存檔。
    用原圖 image_path 取得尺寸 (h,w)，讀取 txt_path 的歸一化多邊形，在黑底上畫白多邊形，
    存成灰階圖到 out_path（評估時與 GT mask 同檔名比對算 IOU/DSC）。
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    labels = _read_txt_labels(txt_path)
    _draw_labels(mask, labels, h, w)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_path, gray)


def _yolo2maskdir_all(
    label_dir: str,
    images_dir: str,
    output_mask_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    將 label_dir 下所有 txt 對應 images_dir 的圖轉成 IOU 用的 mask 寫入 output_mask_dir。
    images_dir：原圖目錄（用於取得尺寸與檔名對應）；label_dir：YOLO 預測輸出的 txt 目錄。
    輸出為與原圖同尺寸的灰階 mask（黑底、預測多邊形為白），檔名與原圖主檔名相同、副檔名 .png。
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp")
    img_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(img_exts)
    ]
    txt_files = [f for f in os.listdir(label_dir) if f.lower().endswith(".txt")]
    prefix = lambda f: os.path.splitext(f)[0]
    img_by_prefix = {prefix(f): f for f in img_files}
    txt_by_prefix = {prefix(f): f for f in txt_files}
    common = set(img_by_prefix) & set(txt_by_prefix)
    for i, base in enumerate(sorted(common)):
        img_name = img_by_prefix[base]
        txt_name = txt_by_prefix[base]
        img_path = os.path.join(images_dir, img_name)
        txt_path = os.path.join(label_dir, txt_name)
        out_name = base + ".png"
        out_path = os.path.join(output_mask_dir, out_name)
        _yolo2mask_one(img_path, txt_path, out_path)
        if progress_callback:
            progress_callback(f"轉換 mask: {img_name}")


def run_analysis(
    model_path: str,
    predict_dir: str,
    gt_dir: str,
    output_csv_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    min_iou: float = 0.5,
    min_dice: float = 0.5,
    output_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    使用訓練好的 YOLO 分割模型對 predict_dir 下所有圖推論，
    與 gt_dir 計算每張 IOU、DSC；低於 min_iou 或 min_dice 的樣本排除於總計，結果寫入 output_csv_path 與 result.txt。
    若指定 output_dir，預測 labels、masks、result.csv、result.txt 將一併寫入該目錄；否則僅輸出 CSV/result.txt（預測用暫存後刪除）。
    回傳 (success, message)。
    """
    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg, flush=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        return False, "請在訓練環境中安裝 ultralytics：pip install ultralytics"

    if not os.path.isfile(model_path):
        return False, f"模型檔案不存在: {model_path}"
    if not os.path.isdir(predict_dir):
        return False, f"Predict 目錄不存在: {predict_dir}"
    if not os.path.isdir(gt_dir):
        return False, f"Ground Truth 目錄不存在: {gt_dir}"

    log("載入模型...")
    model = YOLO(model_path)

    use_output_dir = output_dir and output_dir.strip()
    if use_output_dir:
        work_dir = os.path.abspath(output_dir.strip())
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = None

    def _run_in_work_dir(work_dir: str) -> Tuple[bool, str]:
        # 與 segmentation-and-detection/yolo-segmentation.py 一致：逐張圖片 predict，
        # 確保 save_txt 對每張圖都寫出 txt，並寫入 project/name（predict）目錄
        img_exts = (".png", ".jpg", ".jpeg", ".bmp")
        image_files = [
            f for f in os.listdir(predict_dir)
            if f.lower().endswith(img_exts)
        ]
        if not image_files:
            return False, "Predict 目錄下沒有圖片檔（.png/.jpg/.jpeg/.bmp）。"
        log(f"執行預測（共 {len(image_files)} 張）...")
        predict_dir_abs = os.path.abspath(predict_dir)
        last_results = None
        for i, img_name in enumerate(sorted(image_files)):
            img_path = os.path.join(predict_dir_abs, img_name)
            last_results = model.predict(
                source=img_path,
                save=True,
                save_txt=True,
                project=work_dir,
                name="predict",
                exist_ok=True,
            )
            if (i + 1) % max(1, len(image_files) // 10) == 0:
                log(f"預測進度: {i + 1}/{len(image_files)}")
        # 從 Ultralytics 回傳的 save_dir 取得 labels 路徑（相容 predict 或 segment/predict）
        label_dir = None
        if last_results is not None and len(last_results) > 0:
            try:
                run_save_dir = getattr(last_results[0], "save_dir", None)
                if run_save_dir is not None:
                    run_save_dir = Path(run_save_dir)
                    cand = run_save_dir / "labels"
                    if cand.exists():
                        label_dir = str(cand)
            except Exception:
                pass
        if not label_dir:
            for candidate in (
                Path(work_dir) / "predict" / "labels",
                Path(work_dir) / "segment" / "predict" / "labels",
            ):
                if candidate.exists():
                    label_dir = str(candidate)
                    break
        if not label_dir or not os.path.isdir(label_dir):
            return False, "預測未產生 labels 目錄，請確認模型為分割模型且 Predict 目錄有圖片。"

        mask_dir = os.path.join(work_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        log("將預測 txt 轉為 IOU 用的 mask 圖（與原圖同尺寸、黑底白多邊形）...")
        _yolo2maskdir_all(label_dir, predict_dir, mask_dir, progress_callback=log)
        log(f"IOU 預測用 mask 圖已寫入: {mask_dir}")

        log(f"計算每張圖 IOU、DSC（低於 IOU={min_iou} 或 DSC={min_dice} 者排除於總計）...")
        per_image, total_iou, total_dice = evaluate_miou_mdice_exclude_zero(
            mask_dir, gt_dir, pred_ext=".png", gt_ext=".png",
            min_iou=min_iou, min_dice=min_dice,
        )

        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "IOU", "DSC"])
            for filename, iou, dice in per_image:
                w.writerow([filename, f"{iou:.6f}", f"{dice:.6f}"])
            w.writerow([f"Total (excluding IOU<{min_iou} or DSC<{min_dice})", f"{total_iou:.6f}", f"{total_dice:.6f}"])

        result_txt_path = os.path.join(out_dir, "result.txt") if out_dir else "result.txt"
        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Mean IoU: {total_iou:.6f}\n")
            f.write(f"Mean DSC: {total_dice:.6f}\n")
        log(f"CSV 已寫入: {output_csv_path}")
        log(f"result.txt 已寫入: {result_txt_path}")
        return True, (
            f"分析完成。結果已寫入: {output_csv_path}\n"
            f"result.txt: {result_txt_path}\n"
            f"總 IOU: {total_iou:.4f}, 總 DSC: {total_dice:.4f}"
        )

    try:
        if use_output_dir:
            return _run_in_work_dir(work_dir)
        with tempfile.TemporaryDirectory(prefix="labelme_analysis_") as tmpdir:
            return _run_in_work_dir(tmpdir)
    except Exception as e:
        return False, f"分析過程發生錯誤: {e}"


def _main() -> None:
    parser = argparse.ArgumentParser(description="模型分析：預測 + IOU/DSC 評估，輸出 CSV 與 result.txt")
    parser.add_argument("--model", required=True, help="訓練好的 YOLO 分割模型路徑 (best.pt)")
    parser.add_argument("--predict_dir", required=True, help="要推論的圖片目錄")
    parser.add_argument("--gt_dir", required=True, help="Ground Truth mask 目錄")
    parser.add_argument("--output_csv", required=True, help="輸出 CSV 路徑")
    parser.add_argument("--min_iou", type=float, default=0.5, help="低於此 IOU 的樣本不納入總計（預設 0.5）")
    parser.add_argument("--min_dice", type=float, default=0.5, help="低於此 DSC 的樣本不納入總計（預設 0.5）")
    parser.add_argument("--output_dir", default="", help="可選：分析輸出目錄，預測 labels、masks、result.txt 一併寫入；留空則僅輸出 CSV/result.txt")
    args = parser.parse_args()
    success, msg = run_analysis(
        args.model,
        args.predict_dir,
        args.gt_dir,
        args.output_csv,
        min_iou=args.min_iou,
        min_dice=args.min_dice,
        output_dir=args.output_dir or None,
    )
    if not success:
        print(msg, file=sys.stderr)
        sys.exit(1)
    print(msg)


if __name__ == "__main__":
    _main()
