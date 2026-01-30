"""
YOLO 模型訓練功能模組
支援 YOLOv8 和 YOLOv5 的模型訓練
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple


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
        yolo_version: YOLO 版本 ("v8" 或 "v5")
        
    Returns:
        (is_installed, message)
    """
    try:
        if yolo_version == "v8":
            # 檢查 ultralytics
            result = subprocess.run(
                [python_path, "-c", "import ultralytics; print(ultralytics.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"Ultralytics (YOLOv8) 已安裝: {version}"
            else:
                return False, "Ultralytics (YOLOv8) 未安裝"
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
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv8 (Ultralytics) 訓練模型
    
    Args:
        dataset_yaml: dataset.yaml 文件路徑
        output_dir: 輸出目錄
        python_path: Python 可執行文件路徑
        epochs: 訓練輪數
        imgsz: 圖像大小
        batch: 批次大小
        device: 設備（"0" 為 GPU 0，"cpu" 為 CPU）
        project_name: 專案名稱
        model_size: 模型大小 (n/s/m/l/x)
        progress_callback: 進度回調函數
        
    Returns:
        (success, message)
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 構建訓練腳本（處理 Windows 路徑）
        dataset_yaml_escaped = dataset_yaml.replace('\\', '\\\\')
        output_dir_escaped = output_dir.replace('\\', '\\\\')
        
        train_script = f"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# 設置工作目錄
dataset_yaml_path = r'{dataset_yaml_escaped}'
os.chdir(os.path.dirname(dataset_yaml_path))

# 創建模型
model = YOLO('yolov8{model_size}.pt')

# 開始訓練（workers=0 避免 Windows 下 DataLoader 多進程崩潰）
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
        
        # 將腳本寫入臨時文件
        script_path = os.path.join(output_dir, "train_script.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(train_script)
        
        # 執行訓練
        if progress_callback:
            progress_callback("開始訓練 YOLOv8 模型...")
        
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
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    使用 YOLOv5 訓練模型
    
    Args:
        dataset_yaml: dataset.yaml 文件路徑
        output_dir: 輸出目錄
        python_path: Python 可執行文件路徑
        epochs: 訓練輪數
        imgsz: 圖像大小
        batch: 批次大小
        device: 設備（"0" 為 GPU 0，"cpu" 為 CPU）
        project_name: 專案名稱
        model_size: 模型大小 (n/s/m/l/x)
        progress_callback: 進度回調函數
        
    Returns:
        (success, message)
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 構建訓練命令
        # workers=0 避免 Windows 下 DataLoader 多進程崩潰
        train_cmd = [
            python_path,
            "-m", "yolov5.train",
            "--data", dataset_yaml,
            "--epochs", str(epochs),
            "--img", str(imgsz),
            "--batch", str(batch),
            "--device", device,
            "--workers", "0",
            "--project", output_dir,
            "--name", project_name,
            "--exist-ok",
        ]
        
        if progress_callback:
            progress_callback("開始訓練 YOLOv5 模型...")
        
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
