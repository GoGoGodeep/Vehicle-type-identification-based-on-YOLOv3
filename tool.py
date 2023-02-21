import numpy as np
import torch

def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxs_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 交集的坐标
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])
    w = torch.maximum(torch.tensor(0), xx2 - xx1)
    h = torch.maximum(torch.tensor(0), yy2 - yy1)

    inv = w * h
    if isMin:
        # 交集/最小值
        # 解决大框套小框的问题
        min_area = np.minimum(box_area, boxs_area)
        iou = inv / min_area
    else:
        # 交集/并集
        iou = inv / (box_area + boxs_area - inv)

    return iou


def to_cpu(tensor):
    return tensor.detach().cpu()

def nms(boxes, thresh=0.3, isMin=False):
    # 框的长度为0时(防止程序有缺陷报错)
    if boxes.shape[0] == 0:
        return np.array([])

    # 框的长度不为0时
    # 根据置信度排序：[x1,y1,x2,y2,C]
    _boxes = boxes[(-boxes[:, 4]).argsort()]  # #根据置信度“由大到小”，默认有小到大（加符号可反向排序）
    # 创建空列表，存放保留剩余的框
    r_boxes = []
    # 用1st个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）
    while _boxes.shape[0] > 1:  # shape[0]等价于shape(0),代表0轴上框的个数（维数）
        # 取出第1个框
        a_box = to_cpu(_boxes[0])
        # 取出剩余的框
        b_boxes = to_cpu(_boxes[1:])

        # 将1st个框加入列表
        r_boxes.append(a_box)  ##每循环一次往，添加一个框
        # print(iou(a_box, b_boxes))

        # 比较IOU，将符合阈值条件的的框保留下来
        index = np.where(iou(a_box, b_boxes, isMin) < thresh)  # 将阈值小于0.3的建议框保留下来，返回保留框的索引
        _boxes = b_boxes[index]  # 循环控制条件；取出阈值小于0.3的建议框

    if _boxes.shape[0] > 0:  ##最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0])  # 将此框添加到列表中
    # stack组装为矩阵：:将列表中的数据在0轴上堆叠（行方向）
    r_boxes = np.stack(r_boxes)

    return r_boxes