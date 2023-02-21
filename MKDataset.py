import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision.utils import save_image

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class MKDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(self.path, "images"))
    def __len__(self):
        return len(self.name)
    def __getitem__(self, index):
        name = self.name[index]
        img_path = os.path.join(self.path, "images")
        img = Image.open(os.path.join(img_path, name))

        # 等比缩放
        bg_img = torchvision.transforms.ToPILImage()(torch.zeros(3, 416, 416))

        img_size = torch.Tensor(img.size)
        # 获取最大边长的索引
        l_max_index = img_size.argmax()
        ratio = 416 / img_size[l_max_index]
        img_resize = img_size * ratio
        img_resize = img_resize.long()

        img_use = img.resize(img_resize)

        bg_img.paste(img_use)

        return transform(bg_img), name

if __name__ == '__main__':
    dataset = MKDataset(r"C:\Users\22950\Desktop\car-main\dataset")
    i = 1

    for img, name in dataset:
        print(i)
        save_image(img, r"C:\Users\22950\Desktop\car-main\dataset\data\{0}".format(name), nrow=1)
        i += 1