import numpy as np
import json
import os
import sys
import time
import math
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchattacks.attack import Attack
from utils import *
from compression import *
from decompression import *
from PIL import ImageFile
from cifar10_models import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
from argparse import ArgumentParser

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

class InfoDrop(Attack):
    r"""
    Distance Measure : l_inf  boundon quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, height=32, width=32, steps=40, batch_size=20, block_size=8, q_size=40, q_min=20,
                 q_max=60, trans_type="iff", dir="step.txt", dir_loss="sin_loss.txt", targeted=False, israndom=1,eps_x=2/255):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(InfoDrop, self).__init__("InfoDrop", model)
        self.steps = steps
        self.targeted = targeted
        self.israndom = israndom
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps_x=eps_x
        self.trans_type = trans_type
        self.dir = dir    # 记录step
        self.dir_loss = dir_loss  # 记录loss

        # Value for quantization range      量化范围:
        self.factor_range = [q_min, q_max]

        # Differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0]) / self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size)  # block_n一共分成的块的数量--

        q_ini_table = np.empty((20,3,32,32), dtype=np.float32) # 批量大小，块的数量，，
        q_ini_table.fill(q_size)

        self.q_tables = {"y": torch.from_numpy(q_ini_table)}

    def forward(self, images, labels):

        r"""
        Overridden.
        """
        # print("labels2:", labels)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = torch.tensor(labels, dtype=torch.long)

        adv_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam([self.q_tables["y"]], lr=0.01)  # 优化器

        print("steps的值为：", self.steps)

        x = dwt_8(images)[:, :, 0:16, 0:16].to(self.device)
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            comp = images
            comp = comp.clone().detach().to(self.device)
            x.requires_grad = True

            for k in self.q_tables.keys():
                if (self.trans_type == "dwt"):
                    comp = dwt_8(comp)

                    comp[:, :, 0:16, 0:16] = x

                    comp = quantize(comp, self.q_tables[k], self.alpha)
                    comp = dequantize(comp, self.q_tables[k])
                    comp = idwt_8(comp)

            rgb_images=comp
            rgb_images = rgb_images.type(torch.FloatTensor)
            rgb_images = rgb_images.to(device)

            outputs = self.model(rgb_images)

            _, pre = torch.max(outputs.data, 1)
            labels = labels.to(device)

            x.register_hook(save_grad('x1'))

            if self.targeted:
                suc_rate = (torch.true_divide((pre == labels).sum(), self.batch_size)).cpu().detach().numpy()
            else:  # 无目标
                suc_rate = (torch.true_divide((pre != labels).sum(), self.batch_size)).cpu().detach().numpy()

            adv_cost = adv_loss(outputs, labels)

            if not self.targeted:
                adv_cost = -1 * adv_cost

            total_cost = adv_cost
            optimizer.zero_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval

            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() - torch.sign(self.q_tables[k].grad)  # 更新量化表
                # 随机更新
                if (self.israndom):
                    y = torch.rand(20,3,32,32)
                    y = y * (3 / (i + 1))
                    self.q_tables[k]+= y
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()

                x = x.detach() - torch.sign(grads['x1'])
                x = clip_by_tensor(x, x - eps_x, x + eps_x)


            # if i%10 == 0:
            print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate)
            # # 记录loss
            #
            # filename = os.path.join("./result_data", self.dir_loss)
            # with open(filename, 'a') as f:
            #     f.write('Step:  ' + str(i) + "  Loss:  " + str(total_cost.item()) + "  Current Suc rate: " + str(
            #         suc_rate) + '\n')
            # f.close()

            if suc_rate >= 1:
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                # # 记录loss
                # filename = os.path.join("./result_data", self.dir_loss)
                # with open(filename, 'a') as f:
                #     f.write('End at step  ' + str(i) + "  with suc. rate  " + str(suc_rate) + '\n')
                # f.close()

                # 记录step
                filename = os.path.join("./result_data", self.dir)
                with open(filename, 'a') as f:
                    f.write(str(i+1) + '\n')
                f.close()

                q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
                return q_images, pre, i

        filename = os.path.join("./result_data", self.dir)
        with open(filename, 'a') as f:
            f.write(str(self.steps - 1) + '\n')
        f.close()

        q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
        return q_images, pre, q_table  # 攻击之后图片，攻击后预测标签，攻击后量化表


class Normalize(nn.Module):  # 标准化
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)

def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf = str(round(percentage[i][index[i]].item(), 2))
        pred_list.append([pred_class, pred_conf])
    return pred_list

if __name__ == "__main__":
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_idx = json.load(open("./imagenet_class_index.json"))

    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]  # 标签名
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]  # 文件夹名

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    normalize = transforms.Normalize(mean, std, inplace=False)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # normalize
    ])

    norm_layer = Normalize( mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])

    # Uncomment if you want save results
    save_dir = "./results_image"
    create_dir(save_dir)

    batch_size = 20
    tar_cnt = 1000

    parser = ArgumentParser()
    parser.add_argument("--model", help="resnet50,mobilenet_v2,densenet161", default="resnet50", type=str)
    parser.add_argument("--israndom", default=1, type=int)
    parser.add_argument("--q_size", default=40, type=int)
    parser.add_argument("--eps", default=60, type=int)
    parser.add_argument("--eps_x", default=2/255, type=int)
    parser.add_argument("--trans_type", help="dwt", default="dwt", type=str)
    parser.add_argument("--target", default=0, type=int)
    parser.add_argument("--steps", default=100, type=int)

    args = parser.parse_args()

    eps = args.eps
    trans_type = args.trans_type
    target = args.target
    step1s = args.steps
    q_size = args.q_size
    eps_x=args.eps_x
    model = args.model
    israndom = args.israndom

    q_min = q_size
    q_max = q_size + eps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (model == "resnet50"):
        resnet_model = nn.Sequential(
            norm_layer,
            resnet50(pretrained=True)
        ).to(device)
        resnet_model = resnet_model.eval()

    if (model == "mobilenet_v2"):
        resnet_model = nn.Sequential(
            norm_layer,
            mobilenet_v2(pretrained=True)
        ).to(device)
        resnet_model = resnet_model.eval()

    if (model == "densenet161"):
        resnet_model = nn.Sequential(
            norm_layer,
            densenet161(pretrained=True)
        ).to(device)
        resnet_model = resnet_model.eval()

    dir = ""
    dir_loss = ""

    if (israndom):
        dir = "随机更新" + model + "_" + trans_type + "_" + str(q_size) + "_" + str(q_max - q_size) + "_" + str(
            target) + "_step.txt"
    else:
        dir = model + "_" + trans_type + "_" + str(q_size) + "_" + str(q_max - q_size) + "_" + str(target) + "_step.txt"
    print("dir:", dir)

    if (israndom):
        dir_loss = "随机更新" + model + "_" + trans_type + "_" + str(q_size) + "_" + str(q_max - q_size) + "_" + str(
            target) + "_loss.txt"
    else:
        dir_loss = model + "_" + trans_type + "_" + str(q_size) + "_" + str(q_max - q_size) + "_" + str(
            target) + "_loss.txt"
    print("dir_loss:", dir_loss)


    cur_cnt = 0
    suc_cnt = 0
    data_dir = "./test-data"
    data_clean(data_dir)

    normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)

    # print("normal_data",normal_data)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
    # print(normal_loader)

    normal_iter = iter(normal_loader)

    if (target):
        print("本次进行的是目标攻击")
    else:
        print("本次进行的是无目标攻击")

    if (israndom):
        print("采用随机更新的方法")
    else:
        print("没有用随机更新方法")

    for i in range(tar_cnt // batch_size):  # 50轮
        print("Iter: ", i)
        # filename = os.path.join("./result_data", dir_loss)
        # with open(filename, 'a') as f:
        #     f.write("Iter:   " + str(i) + '\n')
        # f.close()

        images, labels = normal_iter.next()
        # print("images11")
        # print(images[0][0][0])
        # print("本次的图片的labels是：")
        # print(labels)

        # For target attack: set random target.
        # Comment if you set untargeted attack.
        if (target):
            labels = torch.from_numpy(np.random.randint(0, 9, size=batch_size))

        images = images * 255.0
        # print("images22")
        # print(images[0][0][0])

        attack = InfoDrop(resnet_model, batch_size=batch_size, q_size=q_size, steps=step1s, q_max=q_max, q_min=q_min,
                          trans_type=trans_type, dir=dir, dir_loss=dir_loss, targeted=target, israndom=israndom,eps_x=eps_x)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = images.to(device)
        labels = labels.to(device)

        at_images, at_labels, suc_step = attack(images, labels)

        # Uncomment following codes if you want to save the adv imgs  保存图片
        at_images_np = at_images.detach().cpu().numpy()
        adv_img = at_images_np[0]
        # adv_img*=255.0
        adv_img = np.moveaxis(adv_img, 0, 2)

        image_dir = ""
        if (israndom):
            image_dir = "random--" + model + "_" + trans_type + '_' + str(q_size) + '_' + str(
                q_max - q_size) + "_" + str(target)
        else:
            image_dir = model + "_" + trans_type + '_' + str(q_size) + '_' + str(q_max - q_size) + "_" + str(target)

        print("image_dir:", image_dir)
        adv_dir = os.path.join(save_dir, str(image_dir))  # 图像文件名称
        img_name = "adv_{}.png".format(i)
        save_img(adv_img, img_name, adv_dir)

        # 在本轮中 测试20张图片的攻击成功率
        labels = labels.to(device)

        if (target):  # 有目标
            suc_cnt += (at_labels == labels).sum().item()
        else:  # 无目标
            suc_cnt += (at_labels != labels).sum().item()

        print("Current suc. rate: ", suc_cnt / ((i + 1) * batch_size))

        # filename = os.path.join("./result_data", dir_loss)
        # with open(filename, 'a') as f:
        #     f.write("Current suc. rate: " + str(suc_cnt / ((i + 1) * batch_size)) + '\n')
        # f.close()

    score_list = np.zeros(tar_cnt)  # 1000
    score_list[:suc_cnt] = 1.0
    stderr_dist = np.std(np.array(score_list)) / np.sqrt(len(score_list))

    print('Avg suc rate: %.5f +/- %.5f' % (suc_cnt / tar_cnt, stderr_dist))

    filename = os.path.join("./result_data", dir)
    with open(filename, 'a') as f:
        f.write('\n' + str(suc_cnt / tar_cnt) + " +/- " + str(stderr_dist) + '\n')
    f.close()

    suc = suc_cnt / tar_cnt

    # filename = '2.txt'
    # with open(filename, 'a') as f:
    #     f.write(str(suc)+'\n')
    # f.close()