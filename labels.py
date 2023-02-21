import glob
import xml.etree.cElementTree as ET
import numpy as np

ANNOTATIONS_PATH = r"C:\Users\22950\Desktop\car-main\dataset\data\outputs"
CLUSTERS = 9    # 框的数量

# 解析xml文件
def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        for obj in tree.iter("annotation"):
            name = obj.findtext("object/name")
            if name == "bus":
                name = 0
            elif name == "car":
                name = 1
            elif name == "suv":
                name = 2
            elif name == "taxi":
                name = 3
            elif name == "truck":
                name = 4

            xmin = int(obj.findtext("object/bndbox/xmin"))
            ymin = int(obj.findtext("object/bndbox/ymin"))
            xmax = int(obj.findtext("object/bndbox/xmax"))
            ymax = int(obj.findtext("object/bndbox/ymax"))

            cx = int((xmax + xmin) / 2)
            cy = int((ymax + ymin) / 2)
            w = xmax - xmin
            h = ymax - ymin

            filename = obj.findtext("filename")

            dataset.append([filename, name, cx, cy, w, h])

    return np.array(dataset)

if __name__ == '__main__':
    dataset = load_dataset(ANNOTATIONS_PATH)

    filename = "label.txt"
    for i in range(len(dataset)):
        for j in range(6):
            if j != 5:
                with open(filename, 'a', encoding='utf-8') as file:
                    file.write(dataset[i][j] + '\t')
            else:
                with open(filename, 'a', encoding='utf-8') as file:
                    file.write(dataset[i][j] + '\n')

    print("success")