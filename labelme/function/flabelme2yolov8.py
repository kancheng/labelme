import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
  
from sklearn.model_selection import train_test_split
from labelme import utils


class Labelme2YOLO(object):
    
    def __init__(self, json_dir, to_seg=False, other_path=None):
        self._json_dir = json_dir

        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg

        if other_path is None:
            i = 'YOLODataset'
            i += '_seg/' if to_seg else '/'
        else:
            i = other_path + 'YOLODataset'
            i += '_seg/' if to_seg else '/'
        # i = 'YOLODataset'
        # i += '_seg/' if to_seg else '/'
        self._save_path_pfx = os.path.join(self._json_dir, i)

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._save_path_pfx, 'labels/')
        self._image_dir_path = os.path.join(self._save_path_pfx, 'images/')

        for yolo_path in (os.path.join(self._label_dir_path + 'train/'),
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'train/'), 
                          os.path.join(self._image_dir_path + 'val/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            
            os.makedirs(yolo_path)    
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return OrderedDict([(label, label_id) \
                            for label_id, label in enumerate(label_set)])
    
    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        train_idxs, val_idxs = train_test_split(range(len(json_names)), 
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)
        
        self._make_train_val_dir()
    
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/'), 
                                          (train_json_names, val_json_names)):
            for json_name in json_names:
                try:
                    json_path = os.path.join(self._json_dir, json_name)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))
                    img_path = self._save_yolo_image(json_data, 
                                                     json_name, 
                                                     self._image_dir_path, 
                                                     target_dir)
                    yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
                    self._save_yolo_label(json_name, 
                                          self._label_dir_path, 
                                          target_dir, 
                                          yolo_obj_list)
                except Exception as e:
                    print('跳過 %s：%s' % (json_name, e))
        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
                
    def convert_one(self, json_name):
        try:
            json_path = os.path.join(self._json_dir, json_name)
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print('Converting %s ...' % json_name)
            img_path = self._save_yolo_image(json_data, json_name, 
                                             self._json_dir, '')
            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_name, self._json_dir, 
                                  '', yolo_obj_list)
        except Exception as e:
            print('跳過 %s：%s' % (json_name, e))
            raise
    
    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        
        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            
            yolo_obj_list.append(yolo_obj)
            
        return yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                         obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                         for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]

        if self._to_seg:
            retval = [label_id]
            for i in shape['points']:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
        
        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
                    
        yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path, 
                                target_dir, 
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(image_dir_path, target_dir, img_name)

        if not os.path.exists(img_path):
            image_data = json_data.get('imageData')
            if image_data:
                img = utils.img_b64_to_arr(image_data)
                PIL.Image.fromarray(img).save(img_path)
            else:
                # imageData 為 null 時，從 imagePath 讀取外部圖檔
                image_path = json_data.get('imagePath') or img_name
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self._json_dir, image_path)
                image_path = os.path.normpath(image_path)
                if not os.path.exists(image_path):
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                        candidate = os.path.join(self._json_dir, base + ext)
                        if os.path.exists(candidate):
                            image_path = candidate
                            break
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        PIL.Image.fromarray(img_rgb).save(img_path)
                    else:
                        raise FileNotFoundError('Cannot read image: %s' % image_path)
                else:
                    raise FileNotFoundError(
                        'Image not found: %s (imageData is null and imagePath missing or invalid)' % image_path
                    )

        return img_path
    
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._save_path_pfx, 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('path: %s\n' % \
                            os.path.join(self._save_path_pfx.replace("\\", '/'), ''))
            yaml_file.write('train: %s\n' % \
                            os.path.join('images/', 'train'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join('images/', 'val'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)

def lme2yolov8(json_dir, seg, val_size, json_name, other_path=None):

    convertor = Labelme2YOLO(json_dir, to_seg=seg, other_path=other_path)
    if json_name is None:
        convertor.convert(val_size=val_size)
    else:
        convertor.convert_one(json_name)
