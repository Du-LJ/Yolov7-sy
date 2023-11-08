# COCO 格式的数据集转化为 YOLO 格式的数据集
# --json_path 输入的json文件路径
# --save_path 保存的文件夹名字，默认为当前目录下的labels。

import os
import json
from tqdm import tqdm


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    # round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

# 本程序用于将COCO格式的数据集转为YOLO格式
if __name__ == '__main__':
    # 这里根据自己的json文件位置，换成自己的就行
    root = "extractedCoco-dog/"
    json_trainfile = root + 'annotations/instances_train.json'  # COCO Object Instance 类型的标注
    json_valfile = root + 'annotations/instances_val.json'  # COCO Object Instance 类型的标注
    ana_txt_save_path = root + 'labels/'  # 保存的路径

    traindata = json.load(open(json_trainfile, 'r'))
    valdata = json.load(open(json_valfile, 'r'))

    # 重新映射并保存class文件
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
        os.makedirs(ana_txt_save_path + 'train/')
        os.makedirs(ana_txt_save_path + 'val/')

    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(root, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(traindata['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    '''
    保存train txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'train.txt'), 'w')
    for img in tqdm(traindata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, 'train', ana_txt_name), 'w')
        for ann in traindata['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入train的路径
        list_file.write('./images/train/%s.jpg\n' % (head))
    list_file.close()
    '''
    保存val txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'val.txt'), 'w')
    for img in tqdm(valdata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, 'val', ana_txt_name), 'w')
        for ann in valdata['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入val的路径
        list_file.write('./images/val/%s.jpg\n' % (head))
    list_file.close()
