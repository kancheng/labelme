import os, cv2, sys
import numpy as np
from shutil import copy
import argparse

# ├─images
# │  ├─train
# │  └─val
# └─labels
#     ├─train
#     └─val


'''
Read txt annotation files and original images
'''

def read_txt_labels(txt_file):
    """
    Read labels from txt annotation file
    :param txt_file: txt annotation file path
    :return: tag list
    """
    with open(txt_file, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split(" ")
            class_id = int(label_data[0])
            # Parsing bounding box coordinates
            coordinates = [float(x) for x in label_data[1:]]
            labels.append([class_id, coordinates])
    return labels

def draw_labels(mask, labels):
    """
    Draw segmentation regions on the image
    :param image: image
    :param labels: list of labels
    """
    for label in labels:
        class_id, coordinates = label
        # Convert coordinates to integers and reshape into polygons
        points = [(int(x * mask.shape[1]), int(y * mask.shape[0])) for x, y in zip(coordinates[::2], coordinates[1::2])]
        # Use polygon fill
        cv2.fillPoly(mask, [np.array(points)], (255, 255, 255)) # Green indicates segmented area

def yolo2maskdir(kpimg,kptxt,kout):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    # Reading an Image
    # image = cv2.imread("./test/coco128.jpg")
    image = cv2.imread(kpimg)
    height, width, _  = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    # Read txt annotation file
    # txt_file = "./test/coco128.txt"
    txt_file = kptxt
    labels = read_txt_labels(txt_file)
    # Draw segmentation area
    draw_labels(mask, labels)
    # Get the window size
    # window_size = (width//2, height//2) # You can resize the window as needed
    window_size = (width, height) # You can resize the window as needed
    # Resize an image
    mask = cv2.resize(mask, window_size)
    # Create a black image the same size as the window
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    # Place the image in the center of the black background
    mask_x = int((window_size[0] - mask.shape[1]) / 2)
    mask_y = int((window_size[1] - mask.shape[0]) / 2)
    background[mask_y:mask_y + mask.shape[0], mask_x:mask_x + mask.shape[1]] = mask
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # Filename
    # filename = 'savedMasks.jpg'args.output
    filename_o = kout

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_o, mask)


def yolo2masks(txt, img, out):

    ptxt = txt
    # ptxt = "./datasets/yolo/datasetname/labels"
    ptxt_train = ptxt+"/train/"
    ptxt_val = ptxt+"/val/"

    pimg = img
    #pimg = "./datasets/yolo/datasetname/images"
    pimg_train = pimg+"/train/"
    pimg_val = pimg+"/val/"

    pout = out
    #pout = './out/gg'

    pout_mask = pout + "/masks/"
    pout_images = pout + "/images/"

    if not os.path.exists(pout + "/masks/"):
        os.makedirs(pout + "/masks/train")
        os.makedirs(pout + "/masks/val")
    if not os.path.exists(pout + "/images/"):
        os.makedirs(pout + "/images/train")
        os.makedirs(pout + "/images/val")

    def inter(a,b):
        return list(set(a)&set(b))

    ptxt_train_txts = []
    ptxt_val_txts = []
    for filename in os.listdir(ptxt_train):
        if filename.endswith((".txt")):
            ptxt_train_txts.append(filename) 

    for filename in os.listdir(ptxt_val):
        if filename.endswith((".txt")):
            ptxt_val_txts.append(filename)

    pimg_train_imgs = []
    pimg_val_imgs = []
    for filename in os.listdir(pimg_train):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            pimg_train_imgs.append(filename)
    for filename in os.listdir(pimg_val):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            pimg_val_imgs.append(filename)

    for filename in pimg_train_imgs:
        try:
            copy(pimg_train + filename, pout_images + "train/" + filename)
        except Exception as e:
            print(f"跳過複製 train {filename}：{e}")
    for filename in pimg_val_imgs:
        try:
            copy(pimg_val + filename, pout_images + "val/" + filename)
        except Exception as e:
            print(f"跳過複製 val {filename}：{e}")

    ptxt_check=inter(ptxt_train_txts, ptxt_val_txts)
    if ptxt_check:
        print(ptxt_check)
        print(len(ptxt_check))
    else:
        print("TXT Check Empty")

    for num in range(len(pimg_train_imgs)):
        try:
            img_path_train = pimg_train + pimg_train_imgs[num]
            print("img_path_train : ", img_path_train)
            txt_path_train = ptxt_train + ptxt_train_txts[num]
            print("txt_path_train : ", txt_path_train)
            mask_out_path = pout_mask + "train/" + pimg_train_imgs[num]
            print("mask_out_path : ", mask_out_path)
            yolo2maskdir(img_path_train, txt_path_train, mask_out_path)
        except Exception as e:
            print(f"跳過 train {pimg_train_imgs[num]}：{e}")

    for num in range(len(pimg_val_imgs)):
        try:
            img_path_val = pimg_val + pimg_val_imgs[num]
            print("img_path_val : ", img_path_val)
            txt_path_val = ptxt_val + pimg_val_txts[num]
            print("txt_path_val : ", txt_path_val)
            mask_out_path = pout_mask + "val/" + pimg_val_imgs[num]
            print("mask_out_path : ", mask_out_path)
            yolo2maskdir(img_path_val, txt_path_val, mask_out_path)
        except Exception as e:
            print(f"跳過 val {pimg_val_imgs[num]}：{e}")

