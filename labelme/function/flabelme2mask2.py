import os, sys
import json
import numpy as np
import cv2
from PIL import Image
from random import sample

def lme2mask2(input_folder, output_folder):

    if not os.path.exists(input_folder):
        print("No data source.")
        print("沒有數據來源。")
        print("没有数据来源。")
        sys.exit(0)

    print("All Files" , os.listdir(input_folder))

    if not os.path.exists(output_folder + "masks/"):
        os.makedirs(output_folder + "train/masks/")
        os.makedirs(output_folder + "test/masks/")
    if not os.path.exists(output_folder + "images/"):
        os.makedirs(output_folder + "train/images/")
        os.makedirs(output_folder + "test/images/")
    files = []
    jsons = []
    keys = []
    savesubimgname = ""
    for filename in os.listdir(input_folder):
        if filename.endswith((".json")):
            jsons.append(filename)
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp",".JPG")):
            files.append(filename)
            savesubimgname = filename.split(".")[1]

    if len(files) == len(jsons) :
        print("The number of JSON files is the same as the number of image files.")
        print("JSON 檔案的數量與圖片檔案的數量相同。")
        print("JSON 档案的数量与图片档案的数量相同。")
        for filename in jsons:
            keys.append(filename.split(".")[0])
        print(keys)
    else :
        print("The number of JSON files does not match the number of image files, please check.")
        print("JSON 檔案的數量與圖片檔案的數量不一致，請檢查。")
        print("JSON 档案的数量与图片档案的数量不一致，请检查。")
        sys.exit(0)

    test_key = sample(keys, int(len(keys)*0.2))
    train_key = list(set(keys) ^ set(test_key))

    points_save_test = []
    masks_save_test = []
    points_save_train = []
    masks_save_train = []

    # train
    j=0
    for filename in train_key:
        image_path = os.path.join(input_folder, (filename + "." + savesubimgname))
        print("INFO. TRAIN Image Path :", image_path)
        # read image to get shape
        image = cv2.imread(image_path)
        # create a blank image
        mask = np.zeros_like(image, dtype=np.uint8)
        masks_save_train.append(mask)

        points_save_train=[]
        json_path = os.path.join(input_folder, (filename + ".json"))
        print("INFO. TRAIN JSON Path :", json_path)
        with open(json_path, "r") as f:
            data = f.read()
        # convert str to json objs
        data = json.loads(data)
        # get the points
        for i in range(len(data["shapes"])):
            points = data["shapes"][i]["points"]
            points = np.array(points, dtype=np.int32)   # tips: points location must be int32
            points_save_train.append(points)
        for tem in range(len(data["shapes"])):
            cv2.fillPoly(masks_save_train[j], [points_save_train[tem]], (255, 255, 255))
            # save the mask
            output_path = os.path.join(output_folder + "train/masks/", train_key[j] + ".png")
            print("INFO. Output Path :", output_path)
            cv2.imwrite(output_path, masks_save_train[j])
            image_path = os.path.join(input_folder, (train_key[j] + "." + savesubimgname))
            open_image = Image.open(image_path)
            output_path = os.path.join(output_folder + "train/images/", train_key[j] + ".png")
            open_image.save(output_path, "PNG")
        j=j+1

    # test
    j=0
    for filename in test_key:
        image_path = os.path.join(input_folder, (filename + "." + savesubimgname))
        print("INFO. TEST Image Path :", image_path)
        # read image to get shape
        image = cv2.imread(image_path)
        # create a blank image
        mask = np.zeros_like(image, dtype=np.uint8)
        masks_save_test.append(mask)
        points_save_test=[]
        json_path = os.path.join(input_folder, (filename + ".json"))
        print("INFO. TEST JSON Path :", json_path)
        with open(json_path, "r") as f:
            data = f.read()
        # convert str to json objs
        data = json.loads(data)
        # get the points
        for i in range(len(data["shapes"])):
            points = data["shapes"][i]["points"]
            points = np.array(points, dtype=np.int32)   # tips: points location must be int32
            points_save_test.append(points)

        for tem in range(len(data["shapes"])):
            cv2.fillPoly(masks_save_test[j], [points_save_test[tem]], (255, 255, 255))
            # save the mask
            output_path = os.path.join(output_folder + "test/masks/", test_key[j]+".png")
            print("INFO. Output Path :", output_path)
            cv2.imwrite(output_path, masks_save_test[j])
            image_path = os.path.join(input_folder, (test_key[j] + "." + savesubimgname))
            open_image = Image.open(image_path)
            output_path = os.path.join(output_folder + "test/images/", test_key[j]+".png")
            open_image.save(output_path, "PNG")
        j=j+1
