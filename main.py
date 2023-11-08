from PIL import Image
#import matplotlib.pyplot as plt
import torchvision
import os


def picture_size_adjust(file_path):
    print('已开始缩放图片处理')
    folder_path = file_path
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        img = Image.open(folder_path+'\\'+file_name)
        img = torchvision.transforms.Resize((224, 224))(img)
        Image.Image.save(img,'C:\学习资料\项目\sizeadjust\\'+file_name)
    print('图片缩放处理已完成')

def picture_angle_adjust(file_path, angle):
    print('已开始旋转图片处理,旋转角度为正负', angle, '之间')
    folder_path = file_path
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        img = Image.open(folder_path+'\\'+file_name)
        img = torchvision.transforms.RandomRotation(angle)(img)
        Image.Image.save(img,'C:\学习资料\项目\\rotationadjust\\'+file_name)
    print('图片随机旋转操作已完成')

def picture_brightness_adjust(file_path, b1, b2, c1, c2):
    print('开始处理图片亮度和对比度，对比度将调整为原图的',c1,'到',c2,'倍之间，亮度将调整到原图的',b1,'到',b2,'倍之间')
    folder_path = file_path
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        img = Image.open(folder_path+'\\'+file_name)
        img = torchvision.transforms.ColorJitter(brightness=[b1, b2], contrast=[c1, c2])(img)
        Image.Image.save(img, 'C:\学习资料\项目\\brightnessadjust\\'+file_name)
    print('图片亮度以及对比度调整完毕')


if __name__ == '__main__':
    file_path = 'C:\学习资料\项目\新增数据集'
    picture_size_adjust(file_path)
    angle = 90
    #图片随机旋转的角度，可以在任意角度之间进行选择
    picture_angle_adjust(file_path, angle)
    picture_brightness_adjust(file_path, 1,3,1,1.8)

