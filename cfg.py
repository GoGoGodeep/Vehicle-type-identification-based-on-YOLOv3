# 配置文件

# 建议框和图片尺寸
IMG_HEIGHT = 416
IMG_WIDTH = 416
CLASS_NUM = 5

# anchor box是对coco数据集聚类获得的建议框
ANCHOR_GROUP_KMEANS = {
    52: [[10, 13], [16, 30], [33, 23]],
    26: [[30, 61], [62, 45], [59, 119]],
    13: [[116, 90], [156, 198], [373, 326]]
}

# 自定义的建议框
# ANCHOR_GROUP = {
#     13: [[360, 360], [360, 180], [180, 360]],
#     26: [[180, 180], [180, 90], [90, 180]],
#     52: [[90, 90], [90, 45], [45, 90]]      # 尺寸以及对应的建议框（W、H）
# }
ANCHOR_GROUP = {
    13: [[354, 260], [366, 221], [298, 231]],
    26: [[355, 183], [269, 199], [302, 163]],
    52: [[197, 128], [302, 146], [232, 208]]      # 尺寸以及对应的建议框（W、H）
}

# 计算建议框的面积
ANCHOR_GROUP_AREA = {
    13: [x * y for x, y in ANCHOR_GROUP[13]],
    26: [x * y for x, y in ANCHOR_GROUP[26]],
    52: [x * y for x, y in ANCHOR_GROUP[52]],
}

if __name__ == '__main__':
    for feature_size, anchors in ANCHOR_GROUP.items():
        print(feature_size)
        print(anchors)

    for feature_size, anchors_area in ANCHOR_GROUP_AREA.items():
        print(feature_size)
        print(anchors_area)