"""
图片编码
face_dataset/ 存放已知人脸图片
"""

import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface(1)

list_dir = os.listdir("face_dataset")

# 图片路径列表和名字列表一一对应
image_paths = []
names = []
for name in list_dir:
    # 全部图片路径
    image_paths.append("face_dataset/" + name)
    # 存储名字,根据图片下划线前面的人名说明是同一个人
    names.append(name.split("_")[0])

#------------------------------#
#   编码人脸到数据库
#------------------------------#
retinaface.encode_face_dataset(image_paths,names)
