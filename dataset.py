import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
from PIL import Image
import os
import math
import cfg

LABEL_FILE_PATH = r"D:\OneDrive\深度学习\项目\YOLOv3\label.txt"    # 标签数据地址
IMG_BASE_DIR = r"C:\Users\22950\Desktop\car-main\dataset\data"   # 数据总地址

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])  # 对数据进行处理

class dataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()    # 读取标签文件里面的所有数据
            # print(self.dataset)

    def __len__(self):
        return len(self.dataset)    # 获取数据长度

    def __getitem__(self, index):   # 一个一个取出数据
        labels = {}     # 创建一个空的字典，将尺寸大小（13、26、52）作为字典的Key
        line = self.dataset[index]      # 根据索引获取每张图片的数据
        # print(line)

        strs = line.split()     # 将数据分隔开
        # print(strs)
        # ['images/08.jpg', '1', '142', '212', '281', '192', '1', '321', '189', '156', '229']

        # —————————————————————————————————图片处理—————————————————————————————————
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))     # 打开图片数据
        img_data = transforms(_img_data)    # 转成Tensor

        # —————————————————————————————————标签处理—————————————————————————————————
        # _boxes = np.array(float(x) for x in strs[1:])   # 将列表1：后面的所有数据转成float类型
        _boxes = np.array(list(map(float, strs[1:])))   # 也可以使用这种方法

        # 将_boxes列表中的元素5等分，为5是因为每个框有5个标签（置信度、Cx、Cy、W、H）
        # 除以5可以将每个框分开，得到框的数量
        boxes = np.split(_boxes, len(_boxes) // 5)
        # [array([2., 142., 180., 240., 197.]), array([1., 340., 233., 73., 129.])]

        # 循环标签框
        # 13: [[360, 360], [360, 180], [180, 360]]
        for feature_size, anchors in cfg.ANCHOR_GROUP.items():
            # labels[13] = [13, 13, 3（形状）, 5（置信度、Cx、Cy、W、H） + 4（cls_num）]
            # 在空的字典中以feature_size为Key
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM), dtype=np.float32)

            # 循环框的个数
            for box in boxes:
                cls, cx, cy, w, h = box     # 将每个框的数据组成的列表解包赋值给对应变量
                # 计算x、y的偏移量和索引      0.5625, 1.0 = math.modf(50 * 13 / 416)
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_HEIGHT)

                # 循环3种建议框，并带索引分别赋值给i和anchor
                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHOR_GROUP_AREA[feature_size][i]    # 每个建议框的面积
                    # 实际框的高、宽/建议框的高、宽    计算w、h的偏移量
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h      # 实际标签框的面积
                    iou = min(p_area, anchor_area) / max(p_area,anchor_area)    # 计算建议框的实际框的iou

                    labels[feature_size][
                        int(cy_index), int(cx_index), i
                    ] = np.array([
                        iou,
                        cx_offset,
                        cy_offset,
                        np.log(p_w),
                        np.log(p_h),
                        *F.one_hot(
                            torch.tensor(int(cls)), cfg.CLASS_NUM
                        )
                    ])
                    # 总数为5 + cls_num
                    # labels[13] = (13, 13, 3, 9) ————> int(cy_index),int(cx_index)对应索引;i对应形状
                    # 之后根据索引填入对应的区域里面作为标签，表示该区域有目标物体，其他地方没有就为0

        # 返回三种标签和数据
        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    data = dataset()
    dataloader = DataLoader(data, shuffle=True)
    for i, (target_13, target_26, target_52, img_data) in enumerate(dataloader):

        print(img_data.shape)

        exit()