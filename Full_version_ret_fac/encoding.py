import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
# def __init__(self, encoding=0, **kwargs):
retinaface = Retinaface(1)

list_dir = os.listdir("face_dataset")
image_paths = []
names = []
for name in list_dir:
    image_paths.append("face_dataset/"+name)
    # 以"_"为分割,取分割的前半部分
    names.append(name.split("_")[0])

retinaface.encode_face_dataset(image_paths,names)
