import torch.nn as nn
from torch.nn import functional as F

from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1

#---------------------------------------------------#
#   修改模型
#---------------------------------------------------#
class mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MobileNetV1()
        #---------------------------------------------------#
        # 删除无用的池化和全连接
        #---------------------------------------------------#
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        #---------------------------------------------------#
        # 只需要特征提取层
        #---------------------------------------------------#
        x = self.model.stage1(x)    # 160,160,3 -> 20,20,256
        x = self.model.stage2(x)    # 20,20,256 -> 10,10,512
        x = self.model.stage3(x)    # 10,10,512 -> 5,5, 1024
        return x

class inception_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1()

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        """
        embedding_size: 每个人脸用128长度表示
        num_classes:    多少个人,就是多少分类,辅助Triplet Loss的收敛
        """
        super().__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet()
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

        #---------------------------------------------------#
        #   获得一个长度为128的特征向量: 先进行平均池化,再进行全连接层
        #   b,1024,5,5 -> b, 1024,1,1 -> b,1024 -> b,128
        #---------------------------------------------------#
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)

        #---------------------------------------------------#
        #   构建分类器（用于辅助Triplet Loss的收敛）
        #---------------------------------------------------#
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    """预测时使用，直接返回标准化编码结果"""
    def forward(self, x):
        #---------------------------------------------------#
        #   b,3,160,160 -> b,1024,5,5
        #---------------------------------------------------#
        x = self.backbone(x)
        #---------------------------------------------------#
        #   获得一个长度为128的特征向量: 先进行平均池化,再进行全连接层
        #   b,1024,5,5 -> b, 1024,1,1 -> b,1024 -> b,128
        #---------------------------------------------------#
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        #---------------------------------------------------#
        # l2标准化的处理 p=2代表l2
        #---------------------------------------------------#
        x = F.normalize(x, p=2, dim=1)
        # 返回: 标准化前的数据,标准化后的数据
        return x

    """训练时使用，返回标准化前结果和标准化编码结果"""
    def forward_feature(self, x):
        #---------------------------------------------------#
        #   b,3,160,160 -> b,1024,5,5
        #---------------------------------------------------#
        x = self.backbone(x)
        #---------------------------------------------------#
        #   获得一个长度为128的特征向量: 先进行平均池化,再进行全连接层
        #   b,1024,5,5 -> b, 1024,1,1 -> b,1024 -> b,128
        #---------------------------------------------------#
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        #---------------------------------------------------#
        # l2标准化的处理 p=2代表l2
        #---------------------------------------------------#
        x = F.normalize(before_normalize, p=2, dim=1)
        # 返回: 标准化前的数据,标准化后的数据
        return before_normalize, x

    def forward_classifier(self, x):
        """
        x: forward_feature()的before_normalize
        """
        x = self.classifier(x)
        return x
