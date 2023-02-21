import numpy as np
from Net import *
import cfg
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import PIL.ImageDraw as draw
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from tool import nms

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = "cuda"

class Detector(nn.Module):

    def __init__(self, save_path):
        super(Detector, self).__init__()

        self.net = MainNet().to(device)
        self.net.load_state_dict(torch.load(save_path, map_location=device))    # 加载网络参数
        self.net.eval()

    def _filter(self, output, thresh):  # 筛选过滤置信度函数，将置信度合格的留下
        # 数据形状[N, C, H, W] ——> [N, H, W, C]
        output = output.permute(0, 2, 3, 1)

        # 通过reshape变换形状[N, H, W, C] 即 [N, H, W, 45] ——> [N, H, W, 3, 15]
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        # 获取输出置信度大于置信度阈值的目标值的掩码（即布尔值）
        mask = output[..., 0] > thresh
        # 将索引取出来
        idxs = mask.nonzero()
        # 通过掩码获取对应的数据
        vecs = output[mask]

        return idxs, vecs   # 返回索引和数据

    # 定制解析函数，并给4个参数分别是上面筛选合格的框的索引，9个值（中心点偏移和框的偏移以及类别数）
    # t是每个格子的大小（如13*13对应32），t = 总图大小 / 特征图大小，anchor建议框
    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors).to(device)

        # idxs形状（N、V）
        a = idxs[:, 3]  # 表示拿到3个框对应的索引

        # 获取置信度vecs里面有(5+cls_num)个元素，第一个为置信度，使用sigmoid将输出压缩到0-1之间
        confidence = torch.sigmoid(vecs[:, 0])

        # 获取类别数
        _classify = vecs[:, 5:]
        if len(_classify) == 0:     # 获取类别数的长，判断是否为0，避免代码出错
            classify = torch.Tensor([]).to(device)
        else:
            classify = torch.argmax(_classify, dim=1).float()   # 如果不为0，返回类别最大值的索引（类别）

        # idx形状（N, H, W, 3）, vecs（iou, p_x（中心点偏移量）, p_y, p_w（框的偏移量）, p_h, 类别）
        # （索引+偏移量）* 边长 = 中心点坐标
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t     # 计算中心点cy（h+p_y）
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t     # 计算中心点cx（w+p_x）

        # 实际宽、高 = 建议框的宽、高 * e^框的偏移量
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        x1 = cx - w / 2     # 计算框左上角x的坐标
        y1 = cy - h / 2     # 计算框左上角y的坐标
        x2 = x1 + w         # 计算框右下角x的坐标
        y2 = y1 + h         # 计算框右下角y的坐标

        out = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)

        return out

    def forward(self, input, thresh, anchors):  # 传入输入数据、置信度阈值、建议框
        output_13, output_26, output_52 = self.net(input)

        # (n, h, w, 3, 15) 其中n,h,w,3做索引，即这里
        idxs_13, vec_13 = self._filter(output_13, thresh)   # 筛选获取13*13特征图置信度合格的置信度索引和15个值
        boxes_13 = self._parse(idxs_13, vec_13, 32, anchors[13])    # 获得输出框

        idxs_26, vec_26 = self._filter(output_26, thresh)   # 筛选获取26*26特征图置信度合格的置信度索引和15个值
        boxes_26 = self._parse(idxs_26, vec_26, 16, anchors[26])

        idxs_52, vec_52 = self._filter(output_52, thresh)   # 筛选获取52*52特征图置信度合格的置信度索引和15个值
        boxes_52 = self._parse(idxs_52, vec_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)     # 将三种框在0维度按顺序拼接在一起

if __name__ == '__main__':

    save_path = "models/pet.pth"
    img_path = r"E:\OneDrive\DATA\car-main\dataset\img"
    img_name_list = os.listdir(img_path)

    name = {0: 'bus', 1: 'car', 2: 'suv', 3: 'taxi', 4: 'truck'}
    color = {0: 'red', 1: 'red', 2: 'red', 3: 'red', 4: 'red'}

    font = ImageFont.truetype("simsun.ttc", 18, encoding="unic")

    detector = Detector(save_path)

    for image_file in img_name_list:
        im = Image.open(os.path.join(img_path, image_file))

        img = im.convert('RGB')
        img = transforms.ToTensor()(img)
        # 图片的形状为（h, w, c）在0维度加一个轴变为（1, h, w, c）即（n, h, w, c）的形状
        img = img.unsqueeze(0)
        img = img.to(device)

        out_value = detector(img, 0.3, cfg.ANCHOR_GROUP)  # 传入置信度阈值和建议框
        boxes = []  # 定义空列表来装框

        for j in range(5):  # 循环判断类别数
            # 输出的类别如果和类别相同就做NMS去除IOU小的，这里是获取同个类别的掩码
            classify_mask = (out_value[..., -1] == j)
            _boxes = out_value[classify_mask]  # 根据掩码索引对应的输出作为框

            boxes.append(nms(_boxes))# 对同一类别做NMS删掉不合格的框并添加

        for box in boxes:
            try:
                img_draw = draw.ImageDraw(im)  # 制作画笔

                for i in range(len(box)):
                    try:
                        c, x1, y1, x2, y2, cls = box[i, :]  # 解包
                        print(c, x1, y1, x2, y2)
                        # print(int(cls.item()))
                        # print(round(c.item(), 4))     # 取值并保留小数点后4位

                        img_draw.rectangle((x1, y1, x2, y2), outline=color[int(cls.item())], width=2)

                        img_draw.text((max(x1, 0) + 3, max(y1, 0) + 3), fill=color[int(cls.item())],
                                      text=str(int(cls.item())), font=font, width=2)
                        img_draw.text((max(x1, 0) + 15, max(y1, 0) + 3), fill=color[int(cls.item())],
                                      text=name[int(cls.item())], font=font, width=2)
                        img_draw.text((max(x1, 0) + 3, max(y1, 0) + 20), fill=color[int(cls.item())],
                                      text=str(round(c.item(), 2)), font=font, width=2)
                    except:
                        pass
            except:
                continue

        plt.clf()
        plt.ion()
        plt.axis('off')
        plt.imshow(im)
        plt.show()
        plt.pause(3)
        plt.close()
