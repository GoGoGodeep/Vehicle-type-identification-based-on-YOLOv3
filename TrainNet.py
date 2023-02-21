import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import dataset
from Net import MainNet
from torch.utils.tensorboard import SummaryWriter

# 训练
class Trainner:
    def __init__(self):
        self.summayWriter = SummaryWriter("logs")  # 收集训练数据

        # 初始化
        self.save_path = "models/pet.pth"   # 实例化保存的地址
        self.device = torch.device("cuda")
        self.net = MainNet().to(self.device)

        if os.path.exists(self.save_path):  # 判断是否存在之前训练过的网络参数
            self.net.load_state_dict(torch.load(self.save_path))

        # 数据
        self.traindata = dataset()
        self.trainloader = DataLoader(self.traindata, batch_size=8, shuffle=True)

        # 损失函数
        self.conf_loss_fn = nn.BCEWithLogitsLoss()  # 置信度，也可以使用BCELoss，但是需要sigmoid激活
        self.crood_loss_fn = nn.MSELoss()   # 偏移量使用均方差损失
        self.cls_loss_fn = nn.CrossEntropyLoss()    # 标签使用交叉熵损失

        # 优化器
        self.optimzer = optim.Adam(self.net.parameters())   # 定义网络优化器

    # 定义损失函数，传入网络输出的数据、标签和平衡正负样本损失侧重的权重
    def loss_fn(self, output, target, alpha):

        # 将形状转成和标签一样（标签转成和输出形状一样也可以，只要保证形状一样），NCHW——>NHWC
        output = output.permute(0, 2, 3, 1)
        # 通过reshape变换形状 [N, H, W, C] ——> [N, H, W, 3, 15]   如：[3, 13, 13, 3, 15]
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        # 获取正样本，拿到置信度大于0的掩码值返回其索引
        # [..., 0]  ...表示前面不变，在最后的一个维度（15个值）里面取第0个值（置信度）
        mask_obj = target[..., 0] > 0

        output_obj = output[mask_obj]   # 获取输出的正样本
        target_obj = target[mask_obj]   # 获取标签的正样本
        # [:, 0]为置信度，计算置信度损失; [:, 1:5]为Cx、Cy、W、H; [:, 5:]为10分类
        loss_obj_conf = self.conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
        loss_obj_crond = self.crood_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])
        loss_obj_cls = self.cls_loss_fn(output_obj[:, 5:], torch.argmax(target_obj[:, 5:], dim=1))
        # 上方输出的是标量，我们对cls使用了one_hot而交叉熵自带one_hot，需要取最大值
        # 或者使用下面这个，无需对标签取最大值
        # loss_obj_cls = self.conf_loss_fn(output_obj[:, 5:], target_obj[:, 5:])
        loss_obj = loss_obj_crond + loss_obj_cls + loss_obj_conf

        mask_noobj = target[..., 0] == 0    # 获取负样本的掩码
        output_noobj = output[mask_noobj]   # 获得输出的负样本
        target_noobj = target[mask_noobj]   # 获得标签的负样本
        loss_noobj = self.conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])     # 计算负样本损失

        loss = alpha * loss_obj + (1 - alpha) * loss_noobj  # 权重调整正负样本训练程度

        return loss_obj, loss_noobj, loss

    def train(self):
        self.net.train()
        epochs = 0
        while True:
            print(epochs)
            for target_13, target_26, target_52, img_data in self.trainloader:
                target_13, target_26, target_52, img_data = target_13.to(self.device), target_26.to(self.device), target_52.to(self.device), img_data.to(self.device)

                output_13, output_26, output_52 = self.net(img_data)

                loss_13_obj, loss_13_noobj, loss_13 = self.loss_fn(output_13, target_13, 0.9)
                loss_26_obj, loss_26_noobj, loss_26 = self.loss_fn(output_26, target_26, 0.9)
                loss_52_obj, loss_52_noobj, loss_52 = self.loss_fn(output_52, target_52, 0.9)

                loss_obj = loss_13_obj + loss_26_obj + loss_52_obj
                loss_noobj = loss_13_noobj + loss_26_noobj + loss_52_noobj
                loss = loss_13 + loss_26 + loss_52

                self.optimzer.zero_grad()   # 清空梯度
                loss.backward()     # 反向求导
                self.optimzer.step()    # 更新梯度

                # print("正样本损失:", loss_obj.item(), "\t", "负样本损失:", loss_noobj.item(),"\t", "总损失:", loss.item())
                # print(loss_13_noobj, loss_26_noobj, loss_52_noobj)
                print("loss_13:", round(loss_13_obj.item(), 3), "\t",
                      "loss_26:", round(loss_26_obj.item(), 3), "\t",
                      "loss_52:", round(loss_52_obj.item(), 3))

            if epochs % 10 == 0:
                torch.save(self.net.state_dict(), self.save_path.format(epochs))
                self.summayWriter.add_scalars(
                    "loss/acc",
                    {
                        "loss_13": loss_13_obj.item(),
                        "loss_26": loss_26_obj.item(),
                        "loss_52": loss_52_obj.item(),
                        "负样本损失": loss_noobj.item(),
                        "总损失": loss.item()
                    },
                    epochs
                )

            epochs += 1

if __name__ == '__main__':
    obj = Trainner()
    obj.train()