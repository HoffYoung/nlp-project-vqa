import os
# 数据集路径
basic_path = "./data/"
# opencv 人脸检测模型在数据集 mindspore_model_data/opencv_dnn 文件夹中
images_test_path = basic_path + 'images/test'
images_train_path = basic_path + 'images/train'
images_val_path = basic_path + 'images/val'
print(images_test_path)
# 查看文件夹里面文件
os.listdir(images_test_path)