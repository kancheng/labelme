"""
YOLO 模型訓練功能模組
支援 YOLOv8、YOLOv11、YOLOv26 和 YOLOv5 的模型訓練。
預訓練模型可存放於指定目錄，訓練前會先檢查該目錄是否有對應 .pt 檔。
"""

import os
import sys
import yaml
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Ultralytics 預訓練權重下載來源（GitHub assets v8.4.0）
ULTRALYTICS_ASSETS_RELEASE = "v8.4.0"
ULTRALYTICS_ASSETS_BASE = (
    f"https://github.com/ultralytics/assets/releases/download/{ULTRALYTICS_ASSETS_RELEASE}"
)


def get_pretrained_model_path(
    model_dir: Optional[str],
    model_prefix: str,
    model_size: str,
    task: str = "detect",  # "detect" | "segment"
) -> Tuple[str, bool]:
    """
    取得預訓練模型路徑。若指定目錄下存在對應 .pt 檔則回傳本機路徑，否則回傳檔名（訓練時由 Ultralytics 下載）。
    
    Args:
        model_dir: 預訓練模型目錄（可為 None 或空字串）
        model_prefix: 模型前綴，如 "yolov8", "yolo11", "yolo26", "yolov5"
        model_size: 模型大小，如 "n", "s", "m", "l", "x"
        task: 任務類型，"detect" 為檢測，"segment" 為分割（檔名會加 -seg）
    
    Returns:
        (path_for_script, is_local)
        path_for_script: 給訓練腳本用的路徑（本機絕對路徑或單純檔名）
        is_local: 是否為本機已有檔案
    """
    suffix = "-seg" if task == "segment" else ""
    filename = f"{model_prefix}{model_size}{suffix}.pt"
    if model_dir and os.path.isdir(model_dir):
        local_path = os.path.join(model_dir, filename)
        if os.path.isfile(local_path):
            return os.path.abspath(local_path), True
    return filename, False


def download_yolo_pretrained(
    model_dir: str,
    model_prefix: str,
    model_size: str,
    task: str = "detect",  # "detect" | "segment"
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    將 Ultralytics 預訓練權重下載到指定目錄（僅支援 v8/v11/v26 檢測/分割，YOLOv5 需自行準備）。
    
    Args:
        model_dir: 儲存目錄
        model_prefix: 如 "yolov8", "yolo11", "yolo26"
        model_size: 如 "n", "s", "m", "l", "x"
        task: "detect" 或 "segment"
        progress_callback: 可選，progress_callback(message: str) 回報進度
    
    Returns:
        (success, message)
    """
    if model_prefix == "yolov5" and task == "segment":
        if progress_callback:
            progress_callback("YOLOv5 分割權重請自行準備")
        return False, "YOLOv5 分割權重請自行準備，此處僅支援 YOLOv8/v11/v26 下載。"
    suffix = "-seg" if task == "segment" else ""
    filename = f"{model_prefix}{model_size}{suffix}.pt"
    local_path = os.path.join(model_dir, filename)
    if os.path.isfile(local_path):
        if progress_callback:
            progress_callback(f"本機已存在: {filename}")
        return True, f"本機已存在: {local_path}"
    os.makedirs(model_dir, exist_ok=True)
    url = f"{ULTRALYTICS_ASSETS_BASE}/{filename}"
    try:
        if progress_callback:
            progress_callback(f"正在下載: {filename} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunk_size = 8192
            downloaded = 0
            with open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        pct = min(99, int(downloaded * 100 / total))
                        progress_callback(f"下載中 {downloaded // 1024}KB / {total // 1024}KB ({pct}%)")
        if progress_callback:
            progress_callback(f"下載完成: {filename}")
        return True, f"已儲存至: {local_path}"
    except Exception as e:
        if os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return False, f"下載失敗: {str(e)}"


def check_yolo_dataset(dataset_path: str) -> Tuple[bool, str, Dict]:
    """
    檢查 YOLO 數據集格式是否正確
    
    Args:
        dataset_path: YOLO 數據集根目錄路徑
        
    Returns:
        (is_valid, error_message, dataset_info)
        is_valid: 數據集是否有效
        error_message: 錯誤訊息（如果無效）
        dataset_info: 數據集資訊字典
    """
    dataset_path = os.path.abspath(dataset_path)
    
    if not os.path.exists(dataset_path):
        return False, f"數據集路徑不存在: {dataset_path}", {}
    
    # 檢查 dataset.yaml 文件
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if not os.path.exists(yaml_path):
        return False, f"找不到 dataset.yaml 文件: {yaml_path}", {}
    
    # 讀取 dataset.yaml
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
    except Exception as e:
        return False, f"無法讀取 dataset.yaml: {str(e)}", {}
    
    # 檢查必要的配置項
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in dataset_config:
            return False, f"dataset.yaml 缺少必要的配置項: {key}", {}
    
    # 檢查數據目錄結構
    base_path = dataset_config.get('path', dataset_path)
    if not os.path.isabs(base_path):
        base_path = os.path.join(dataset_path, base_path)
    
    train_images = os.path.join(base_path, dataset_config['train'])
    val_images = os.path.join(base_path, dataset_config['val'])
    
    # 檢查 train 和 val 目錄
    train_labels = train_images.replace('images', 'labels')
    val_labels = val_images.replace('images', 'labels')
    
    errors = []
    
    if not os.path.exists(train_images):
        errors.append(f"訓練圖像目錄不存在: {train_images}")
    elif len(os.listdir(train_images)) == 0:
        errors.append(f"訓練圖像目錄為空: {train_images}")
    
    if not os.path.exists(train_labels):
        errors.append(f"訓練標籤目錄不存在: {train_labels}")
    elif len(os.listdir(train_labels)) == 0:
        errors.append(f"訓練標籤目錄為空: {train_labels}")
    
    if not os.path.exists(val_images):
        errors.append(f"驗證圖像目錄不存在: {val_images}")
    elif len(os.listdir(val_images)) == 0:
        errors.append(f"驗證圖像目錄為空: {val_images}")
    
    if not os.path.exists(val_labels):
        errors.append(f"驗證標籤目錄不存在: {val_labels}")
    elif len(os.listdir(val_labels)) == 0:
        errors.append(f"驗證標籤目錄為空: {val_labels}")
    
    if errors:
        return False, "；".join(errors), {}
    
    # 統計數據集資訊
    train_img_count = len([f for f in os.listdir(train_images) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    val_img_count = len([f for f in os.listdir(val_images) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    train_label_count = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
    val_label_count = len([f for f in os.listdir(val_labels) if f.endswith('.txt')])
    
    dataset_info = {
        'yaml_path': yaml_path,
        'base_path': base_path,
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'num_classes': dataset_config['nc'],
        'class_names': dataset_config['names'],
        'train_images_count': train_img_count,
        'train_labels_count': train_label_count,
        'val_images_count': val_img_count,
        'val_labels_count': val_label_count,
    }
    
    return True, "", dataset_info


def check_yolo_installation(python_path: str, yolo_version: str = "v8") -> Tuple[bool, str]:
    """
    檢查 YOLO 是否已安裝在指定的 Python 環境中
    
    Args:
        python_path: Python 可執行文件路徑
        yolo_version: YOLO 版本 ("v8", "v11", "v26" 用 Ultralytics；"v5" 用 yolov5)
        
    Returns:
        (is_installed, message)
    """
    try:
        if yolo_version in ("v8", "v11", "v26"):
            # 檢查 ultralytics（YOLOv8 / v11 / v26 皆為 Ultralytics）
            result = subprocess.run(
                [python_path, "-c", "import ultralytics; print(ultralytics.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                name = {"v8": "YOLOv8", "v11": "YOLOv11", "v26": "YOLOv26"}.get(yolo_version, "YOLO")
                return True, f"Ultralytics ({name}) 已安裝: {version}"
            else:
                return False, f"Ultralytics ({yolo_version.upper()}) 未安裝"
        else:  # v5
            # 檢查 yolov5
            result = subprocess.run(
                [python_path, "-c", "import yolov5; print(yolov5.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"YOLOv5 已安裝: {version}"
            else:
                return False, "YOLOv5 未安裝"
    except subprocess.TimeoutExpired:
        return False, "檢查超時"
    except Exception as e:
        return False, f"檢查時發生錯誤: {str(e)}"


def train_yolo_v8(
    dataset_yaml: str,
    output_dir: str,
    python_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project_name: str = "yolov8_training",
    model_size: str = "n",  # n, s, m, l, x
    model_dir: Optional[str] = None,
    task: str = "detect",  # "detect" | "segment"
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv8 (Ultralytics) 訓練模型。若 model_dir 內有對應 .pt 則使用本機模型，否則從網路下載。
    task: "detect" 檢測、"segment" 分割。
    """
    return _train_yolo_ultralytics(
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        python_path=python_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project_name=project_name,
        model_size=model_size,
        model_dir=model_dir,
        task=task,
        progress_callback=progress_callback,
        model_prefix="yolov8",
        display_name="YOLOv8",
    )


def train_yolo_v11(
    dataset_yaml: str,
    output_dir: str,
    python_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project_name: str = "yolo11_training",
    model_size: str = "n",  # n, s, m, l, x
    model_dir: Optional[str] = None,
    task: str = "detect",  # "detect" | "segment"
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv11 (Ultralytics) 訓練模型。若 model_dir 內有對應 .pt 則使用本機模型。
    task: "detect" 檢測、"segment" 分割。
    """
    return _train_yolo_ultralytics(
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        python_path=python_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project_name=project_name,
        model_size=model_size,
        model_dir=model_dir,
        task=task,
        progress_callback=progress_callback,
        model_prefix="yolo11",
        display_name="YOLOv11",
    )


def train_yolo_v26(
    dataset_yaml: str,
    output_dir: str,
    python_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project_name: str = "yolo26_training",
    model_size: str = "n",  # n, s, m, l, x
    model_dir: Optional[str] = None,
    task: str = "detect",  # "detect" | "segment"
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv26 (Ultralytics) 訓練模型。若 model_dir 內有對應 .pt 則使用本機模型。
    task: "detect" 檢測、"segment" 分割。
    """
    return _train_yolo_ultralytics(
        dataset_yaml=dataset_yaml,
        output_dir=output_dir,
        python_path=python_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project_name=project_name,
        model_size=model_size,
        model_dir=model_dir,
        task=task,
        progress_callback=progress_callback,
        model_prefix="yolo26",
        display_name="YOLOv26",
    )


def _train_yolo_ultralytics(
    dataset_yaml: str,
    output_dir: str,
    python_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project_name: str,
    model_size: str,
    model_dir: Optional[str],
    task: str,  # "detect" | "segment"
    progress_callback: Optional[callable],
    model_prefix: str,  # "yolov8", "yolo11", "yolo26"
    display_name: str,
) -> Tuple[bool, str]:
    """Ultralytics YOLO (v8/v11/v26) 共用訓練邏輯。訓練前先檢查 model_dir 是否有本機模型。"""
    try:
        model_path, is_local = get_pretrained_model_path(
            model_dir, model_prefix, model_size, task=task
        )
        model_path_escaped = model_path.replace('\\', '\\\\').replace("'", "\\'")
        os.makedirs(output_dir, exist_ok=True)
        dataset_yaml_escaped = dataset_yaml.replace('\\', '\\\\')
        output_dir_escaped = output_dir.replace('\\', '\\\\')
        train_script = f"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

dataset_yaml_path = r'{dataset_yaml_escaped}'
os.chdir(os.path.dirname(dataset_yaml_path))

model_path = r'{model_path_escaped}'
model = YOLO(model_path)

results = model.train(
    data=dataset_yaml_path,
    epochs={epochs},
    imgsz={imgsz},
    batch={batch},
    device='{device}',
    project=r'{output_dir_escaped}',
    name='{project_name}',
    workers=0,
    exist_ok=True,
    verbose=True,
)

print("訓練完成！")
print(f"最佳模型保存在: {{results.save_dir}}")
"""
        script_path = os.path.join(output_dir, "train_script.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(train_script)
        if progress_callback:
            if is_local:
                progress_callback(f"使用本機預訓練模型: {model_path}")
            else:
                progress_callback(f"未找到本機模型，訓練時將從網路下載（需連網）")
            task_label = "分割" if task == "segment" else "檢測"
            progress_callback(f"開始訓練 {display_name} {task_label} 模型...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            [python_path, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=output_dir,
            env=env,
        )
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                if progress_callback:
                    progress_callback(line)
        process.wait()
        if process.returncode == 0:
            best_model_path = os.path.join(output_dir, project_name, "weights", "best.pt")
            if os.path.exists(best_model_path):
                return True, f"訓練成功！模型保存在: {best_model_path}"
            return True, f"訓練完成，但未找到最佳模型文件。輸出目錄: {os.path.join(output_dir, project_name)}"
        error_msg = "\n".join(output_lines[-10:])
        return False, f"訓練失敗:\n{error_msg}"
    except Exception as e:
        return False, f"訓練過程中發生錯誤: {str(e)}"


def train_yolo_v5(
    dataset_yaml: str,
    output_dir: str,
    python_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project_name: str = "yolov5_training",
    model_size: str = "n",  # n, s, m, l, x
    model_dir: Optional[str] = None,
    task: str = "detect",  # "detect" | "segment"
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv5 訓練模型。task: "detect" 檢測、"segment" 分割（權重需自行準備 yolov5n-seg.pt 等）。
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        weights_path, is_local = get_pretrained_model_path(
            model_dir, "yolov5", model_size, task=task
        )
        # 構建訓練命令；workers=0 避免 Windows 下 DataLoader 多進程崩潰
        train_cmd = [
            python_path,
            "-m", "yolov5.train",
            "--data", dataset_yaml,
            "--weights", weights_path,
            "--epochs", str(epochs),
            "--img", str(imgsz),
            "--batch", str(batch),
            "--device", device,
            "--workers", "0",
            "--project", output_dir,
            "--name", project_name,
            "--exist-ok",
        ]
        if task == "segment":
            train_cmd.extend(["--task", "segment"])
        if progress_callback:
            task_label = "分割" if task == "segment" else "檢測"
            progress_callback(f"開始訓練 YOLOv5 {task_label} 模型...")
            if is_local:
                progress_callback(f"使用本機預訓練模型: {weights_path}")
        
        # 執行訓練（encoding=utf-8 避免 Windows GBK 解碼錯誤）
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=output_dir,
            env=env,
        )
        
        # 實時輸出
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                if progress_callback:
                    progress_callback(line)
        
        process.wait()
        
        if process.returncode == 0:
            # 查找最佳模型
            best_model_path = os.path.join(output_dir, project_name, "weights", "best.pt")
            if os.path.exists(best_model_path):
                return True, f"訓練成功！模型保存在: {best_model_path}"
            else:
                return True, f"訓練完成，但未找到最佳模型文件。輸出目錄: {os.path.join(output_dir, project_name)}"
        else:
            error_msg = "\n".join(output_lines[-10:])  # 最後10行錯誤訊息
            return False, f"訓練失敗:\n{error_msg}"
            
    except Exception as e:
        return False, f"訓練過程中發生錯誤: {str(e)}"
