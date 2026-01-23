"""
PyTorch 模型轉換為 ONNX 格式功能模組
支援將 .pt 格式的 YOLO 模型轉換為 .onnx 格式
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def check_torch_installation(python_path: str) -> Tuple[bool, str]:
    """
    檢查 PyTorch 是否已安裝在指定的 Python 環境中
    
    Args:
        python_path: Python 可執行文件路徑
        
    Returns:
        (is_installed, message)
    """
    try:
        # 檢查 torch
        result = subprocess.run(
            [python_path, "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"PyTorch 已安裝: {version}"
        else:
            return False, "PyTorch 未安裝"
    except subprocess.TimeoutExpired:
        return False, "檢查超時"
    except Exception as e:
        return False, f"檢查時發生錯誤: {str(e)}"


def check_ultralytics_installation(python_path: str) -> Tuple[bool, str]:
    """
    檢查 Ultralytics (YOLOv8) 是否已安裝
    
    Args:
        python_path: Python 可執行文件路徑
        
    Returns:
        (is_installed, message)
    """
    try:
        result = subprocess.run(
            [python_path, "-c", "import ultralytics; print(ultralytics.__version__)"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Ultralytics 已安裝: {version}"
        else:
            return False, "Ultralytics 未安裝"
    except subprocess.TimeoutExpired:
        return False, "檢查超時"
    except Exception as e:
        return False, f"檢查時發生錯誤: {str(e)}"


def convert_pt_to_onnx(
    model_path: str,
    output_path: str,
    python_path: str,
    imgsz: int = 640,
    simplify: bool = True,
    opset: int = 12,
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """
    將 PyTorch (.pt) 模型轉換為 ONNX (.onnx) 格式
    
    Args:
        model_path: 輸入 .pt 模型文件路徑
        output_path: 輸出 .onnx 文件路徑
        python_path: Python 可執行文件路徑
        imgsz: 輸入圖像大小（默認 640）
        simplify: 是否簡化 ONNX 模型（默認 True）
        opset: ONNX opset 版本（默認 12）
        progress_callback: 進度回調函數
        
    Returns:
        (success, message)
    """
    try:
        # 檢查輸入文件
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        if not model_path.lower().endswith('.pt'):
            return False, f"輸入文件不是 .pt 格式: {model_path}"
        
        # 確保輸出目錄存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 檢查依賴
        if progress_callback:
            progress_callback("正在檢查依賴...")
        
        is_torch_installed, torch_msg = check_torch_installation(python_path)
        if not is_torch_installed:
            return False, f"PyTorch 未安裝: {torch_msg}"
        
        is_ultralytics_installed, ultralytics_msg = check_ultralytics_installation(python_path)
        if not is_ultralytics_installed:
            return False, f"Ultralytics 未安裝: {ultralytics_msg}"
        
        if progress_callback:
            progress_callback(f"✓ {torch_msg}")
            progress_callback(f"✓ {ultralytics_msg}")
        
        # 構建轉換腳本（處理 Windows 路徑）
        model_path_escaped = model_path.replace('\\', '\\\\')
        output_path_escaped = output_path.replace('\\', '\\\\')
        
        convert_script = f"""
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# 設置標準輸出編碼為 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ultralytics import YOLO

# 載入模型
model_path = r'{model_path_escaped}'
print(f"正在載入模型: {{model_path}}")

try:
    model = YOLO(model_path)
    print("模型載入成功")
except Exception as e:
    print(f"模型載入失敗: {{e}}")
    sys.exit(1)

# 轉換為 ONNX
output_path = r'{output_path_escaped}'
print(f"正在轉換為 ONNX: {{output_path}}")

try:
    # 執行轉換，export 方法會返回導出文件的路徑
    exported_path = model.export(
        format='onnx',
        imgsz={imgsz},
        simplify={repr(simplify)},
        opset={opset},
        verbose=True,
    )
    
    # exported_path 可能是字符串或 Path 對象
    if hasattr(exported_path, '__str__'):
        exported_path = str(exported_path)
    
    # 如果導出的文件存在，移動到指定位置
    if exported_path and os.path.exists(exported_path):
        import shutil
        # 如果目標路徑與導出路徑相同，則不需要移動
        if os.path.abspath(exported_path) != os.path.abspath(output_path):
            shutil.move(exported_path, output_path)
            print(f"轉換成功！模型已保存至: {{output_path}}")
        else:
            print(f"轉換成功！模型已保存至: {{output_path}}")
    else:
        # 如果 export 沒有返回路徑，嘗試在模型同目錄下查找
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        possible_paths = [
            os.path.join(model_dir, f"{{model_name}}.onnx"),
            os.path.join(model_dir, f"{{model_name}}_onnx.onnx"),
        ]
        found = False
        for possible_path in possible_paths:
            if os.path.exists(possible_path):
                import shutil
                if os.path.abspath(possible_path) != os.path.abspath(output_path):
                    shutil.move(possible_path, output_path)
                print(f"轉換成功！模型已保存至: {{output_path}}")
                found = True
                break
        
        if not found:
            print(f"警告：無法找到導出的 ONNX 文件")
            if exported_path:
                print(f"export 返回的路徑: {{exported_path}}")
            print(f"預期位置: {{possible_paths[0]}}")
            sys.exit(1)
            
except Exception as e:
    print(f"轉換失敗: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # 將腳本寫入臨時文件
        script_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else os.getcwd()
        script_path = os.path.join(script_dir, "convert_to_onnx_temp.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(convert_script)
        
        if progress_callback:
            progress_callback("開始轉換模型...")
        
        # 執行轉換
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        
        process = subprocess.Popen(
            [python_path, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # 遇到無法解碼的字符時替換為替代字符，而不是拋出錯誤
            bufsize=1,
            universal_newlines=True,
            cwd=os.path.dirname(model_path),
            creationflags=creationflags,
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
        
        # 清理臨時腳本
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except:
            pass
        
        if process.returncode == 0:
            if os.path.exists(output_path):
                return True, f"轉換成功！模型已保存至: {output_path}"
            else:
                return False, f"轉換完成，但找不到輸出文件: {output_path}"
        else:
            error_msg = "\n".join(output_lines[-10:])  # 最後10行錯誤訊息
            return False, f"轉換失敗:\n{error_msg}"
            
    except Exception as e:
        return False, f"轉換過程中發生錯誤: {str(e)}"
