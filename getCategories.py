import os
import json
import shutil

# 程序用于从COCO中提取类别并另存为COCO格式的数据
# 定义要提取的类别
categories = ['dog']
# 定义数据集路径
data_dir = './'
# 定义输出路径
output_dir = './extractedCoco-dog'


if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'annotations'))
        os.makedirs(os.path.join(output_dir, 'images', 'train'))
        os.makedirs(os.path.join(output_dir, 'images', 'val'))
        os.makedirs(os.path.join(output_dir, 'images', 'test'))

    '''
    训练集
    '''
    # 加载原始instances文件
    with open(os.path.join(data_dir, 'annotations', 'instances_train2017.json'), 'r') as f:
        train_instances = json.load(f)

    # 筛选目标类别的id
    target_ids = []
    new_categories = []

    # 类别字典：
    # category_id：category_name, image_number
    categories_dict = {}
    for c in train_instances['categories']:
        if c['name'] in categories:
            target_ids.append(c['id'])
            new_categories.append(c)
            categories_dict[c['id']] = [c['name'], 0]

    # 筛选出训练集中包含目标类别的图片id
    train_image_ids = set()
    new_train_annotations = []
    for ann in train_instances['annotations']:
        if ann['category_id'] in target_ids:
            train_image_ids.add(ann['image_id'])
            new_train_annotations.append(ann)
            categories_dict[ann['category_id']][1] += 1

    new_images = []

    # 复制训练集中包含目标类别的图片到输出目录
    for image in train_instances['images']:
        if image['id'] in train_image_ids:
            new_images.append(image)
            shutil.copy(os.path.join(data_dir, 'train2017', image['file_name']),
                        os.path.join(output_dir, 'images', 'train'))

    # 构造新的instances文件
    new_train_instances = {
        'info': train_instances['info'],
        'licenses': train_instances['licenses'],
        'images': new_images,
        'annotations': new_train_annotations,
        'categories': new_categories
    }

    # 保存新的instances文件
    with open(os.path.join(output_dir, 'annotations', 'instances_train.json'), 'w') as f:
        json.dump(new_train_instances, f)

    print("训练集中包括：")
    for category, number in categories_dict.values():
        print(category + ' ' + str(number) + "张")

    # 加载原始instances文件
    with open(os.path.join(data_dir, 'annotations', 'instances_val2017.json'), 'r') as f:
        val_instances = json.load(f)

    # 计数
    categories_dict = {}
    for c in val_instances['categories']:
        if c['name'] in categories:
            categories_dict[c['id']] = [c['name'], 0]

    # 筛选出验证集中包含目标类别的图片id
    val_image_ids = set()
    new_val_annotations = []
    for ann in val_instances['annotations']:
        if ann['category_id'] in target_ids:
            val_image_ids.add(ann['image_id'])
            new_val_annotations.append(ann)
            categories_dict[ann['category_id']][1] += 1

    new_images = []

    # 复制验证集中包含目标类别的图片到输出目录
    for image in val_instances['images']:
        if image['id'] in val_image_ids:
            new_images.append(image)
            shutil.copy(os.path.join(data_dir, 'val2017', image['file_name']),
                        os.path.join(output_dir, 'images', 'val'))

    # 构造新的instances文件
    new_val_instances = {
        'info': val_instances['info'],
        'licenses': val_instances['licenses'],
        'images': new_images,
        'annotations': new_val_annotations,
        'categories': new_categories
    }

    # 保存新的instances文件
    with open(os.path.join(output_dir, 'annotations', 'instances_val.json'), 'w') as f:
        json.dump(new_val_instances, f)

    print("验证集中包括：")
    for category, number in categories_dict.values():
        print(category + ' ' + str(number) + "张")
