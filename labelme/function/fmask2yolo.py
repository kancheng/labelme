
# ├─mask_dataset1
# │  ├─train
# │  │  ├─images
# │  │  └─masks
# │  └─val
# │      ├─images
# │      └─masks
# └─mask_dataset2
#     ├─images
#     │  ├─train
#     │  └─val
#     └─masks
#         ├─train
#         └─val
import copy
import cv2
import os
import shutil
import numpy as np

def eachmask2yolo(path, save_path, procshow=False):
    files = os.listdir(path)
    for file in files:
        try:
            name = file.split('.')[0]
            file_path = os.path.join(path, name + '.png')
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"無法讀取圖像: {file_path}")
            H, W = img.shape[0:2]

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnt, hit = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

            cnt = list(cnt)
            f = open(save_path + f"/{name}.txt", "a+")
            for j in cnt:
                result = []
                pre = j[0]
                for i in j:
                    if abs(i[0][0] - pre[0][0]) > 1 or abs(i[0][1] - pre[0][1]) > 1:
                        pre = i
                        temp = list(i[0])
                        temp[0] = float(temp[0]) / W
                        temp[1] = float(temp[1]) / H
                        result.append(temp)

                        if procshow:
                            cv2.circle(img, i[0], 1, (0, 0, 255), 2)

                if len(result) != 0:
                    f.write("0 ")
                    for line in result:
                        line = " ".join(map(str, line))
                        f.write(line + " ")
                    f.write("\n")
            f.close()
        except Exception as e:
            print(f"跳過 {file}：{e}")
            continue

        if procshow:
            cv2.imshow("test", img)
            while True:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()

def mask2yolo(input_dir, output_dir):
    def is_mask_dataset1(path):
        return os.path.exists(os.path.join(path, "train", "images")) and os.path.exists(os.path.join(path, "train", "masks"))

    def is_mask_dataset2(path):
        return os.path.exists(os.path.join(path, "images", "train")) and os.path.exists(os.path.join(path, "masks", "train"))

    if is_mask_dataset1(input_dir):
        for split in ["train", "val"]:
            image_src_dir = os.path.join(input_dir, split, "images")
            mask_src_dir = os.path.join(input_dir, split, "masks")
            image_dst_dir = os.path.join(output_dir, "images", split)
            label_dst_dir = os.path.join(output_dir, "labels", split)

            os.makedirs(image_dst_dir, exist_ok=True)
            os.makedirs(label_dst_dir, exist_ok=True)

            for img_file in os.listdir(image_src_dir):
                try:
                    shutil.copy(os.path.join(image_src_dir, img_file), image_dst_dir)
                except Exception as e:
                    print(f"跳過複製 {img_file}：{e}")

            eachmask2yolo(mask_src_dir, label_dst_dir)

    elif is_mask_dataset2(input_dir):
        for split in ["train", "val"]:
            image_src_dir = os.path.join(input_dir, "images", split)
            mask_src_dir = os.path.join(input_dir, "masks", split)
            image_dst_dir = os.path.join(output_dir, "images", split)
            label_dst_dir = os.path.join(output_dir, "labels", split)

            os.makedirs(image_dst_dir, exist_ok=True)
            os.makedirs(label_dst_dir, exist_ok=True)

            for img_file in os.listdir(image_src_dir):
                try:
                    shutil.copy(os.path.join(image_src_dir, img_file), image_dst_dir)
                except Exception as e:
                    print(f"跳過複製 {img_file}：{e}")

            eachmask2yolo(mask_src_dir, label_dst_dir)

    else:
        raise ValueError("Input directory structure does not match any supported dataset format.")

    dataset_yaml_content = f"""
    train: ./images/train
    val: ./images/val

    nc: 1
    names: ['object']
    """
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as yaml_file:
        yaml_file.write(dataset_yaml_content)
