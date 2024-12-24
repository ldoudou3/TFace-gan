import os
import sys
import models
import torch
import torch.fft
import wandb
import time
import numpy as np
import torch.cuda.amp as amp
import torch.optim as optim
import hashlib
from torch.nn.parallel import DistributedDataParallel


# from torchjpeg import dct
# from torch.nn import functional as F
from scipy.spatial import ConvexHull, Delaunay

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather
from torchkit.loss import get_loss
from torchkit.task import BaseTask

# import random

class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def loop_step(self, epoch):
        """
        load_data
            |
        extract feature
            |
        optimizer step
            |
        print log and write summary
        """
        backbone, heads = self.backbone, list(self.heads.values())
        discriminator = self.backbone_discriminator # D

        backbone.train()  # set to training mode 将主干模型设置为训练模式。
        discriminator.train() # D

        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for step, samples in enumerate(self.train_loader):
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts = self.opt['backbone'], list(self.opt['heads'].values())
            discriminator_opt  = self.opt_discriminator['discriminator'] # D

            # 准备输入数据和标签。
            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)

            # train
            if self.amp:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)


            # gather features
            features_gather = AllGather(features, self.world_size)
            features_gather = [torch.split(x, batch_sizes) for x in features_gather]
            all_features = []
            for i in range(len(batch_sizes)):
                all_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather labels
            with torch.no_grad():
                labels_gather = AllGather(labels, self.world_size)
            labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
            all_labels = []
            for i in range(len(batch_sizes)):
                all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            # gather inputs
            inputs_gather = AllGather(inputs, self.world_size)
            inputs_gather = [torch.split(x, batch_sizes) for x in inputs_gather]
            all_inputs = []
            for i in range(len(batch_sizes)):
                all_inputs.append(torch.cat([x[i] for x in inputs_gather], dim=0).cuda())
       

            losses_generator = [] # 生成器总损失
            losses_fr = [] # 识别
            losses_anta = [] # 对抗
            losses_discriminator = [] # 判别器损失
            for i in range(len(batch_sizes)):
                # PartialFC need update optimizer state in training process
                if self.pfc:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i])

                loss_fr = self.loss(outputs, labels) * self.branch_weights[i] # 识别损失
                loss_anta = self.anta_loss(all_features[i],discriminator) # 对抗损失
                loss_generator =  loss_fr + loss_anta

                losses_fr.append(loss_fr)
                losses_anta.append(loss_anta)
                losses_generator.append(loss_generator)

                # 生成噪声向量
                noise_embed = self.generate_fixed_noise(all_inputs[i], 512) # D
                discriminator_loss  = self.discriminator_loss(noise_embed,all_features[i].detach().clone(),discriminator,epoch) # D
                losses_discriminator.append(discriminator_loss) # D
                
                prec1, prec5 = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             all_labels[i],
                                             self.class_shards[i],
                                             topk=(1, 5))
                am_losses[i].update(loss_generator.data.item(), all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))

            # update summary and log_buffer
            scalars = {
                'train/loss': am_losses,
                'train/top1': am_top1s,
                'train/top5': am_top5s,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'loss': am_losses,
                'prec@1': am_top1s,
                'prec@5': am_top5s,
            }
            self.update_log_buffer(log)

            # compute loss
            total_loss_fr = sum(losses_fr)
            total_loss_anta = sum(losses_anta)
            total_loss_generator = sum(losses_generator)
            total_loss_discriminator = sum(losses_discriminator) # D    

            wandb.log({
                    "loss_generator": total_loss_generator,
                    "loss_discriminator": total_loss_discriminator, #D
                    "loss_generator_fr":total_loss_fr,
                    "loss_generator_anta":total_loss_anta
                }, step=step + epoch * len(self.train_loader))
            
            # compute gradient and do SGD
            backbone_opt.zero_grad()
            discriminator_opt.zero_grad() # D
            for head_opt in head_opts:
                head_opt.zero_grad()


            # Automatic Mixed Precision setting
            if self.amp:
                self.scaler.scale(total_loss_generator).backward(retain_graph=True)
                self.scaler.step(backbone_opt)

                self.scaler.scale(total_loss_discriminator).backward(retain_graph=True) # D
                self.scaler.step(discriminator_opt) # D

                for head_opt in head_opts:
                    self.scaler.step(head_opt)
                self.scaler.update()
            else:
                total_loss_generator.backward(retain_graph=True)
                backbone_opt.step()

                total_loss_discriminator.backward(retain_graph=True) # D
                discriminator_opt.step() # D

                for head_opt in head_opts:
                    head_opt.step()

            # PartialFC need update weight and weight_norm manually
            if self.pfc:
                for head in heads:
                    head.update()

            cost = t.get_duration()

            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)
    
    def get_discriminator_optimizer(self):
        """ build optimizers for discriminator """
        # 从配置中读取超参数
        init_lr = self.cfg['DISCRIMINATOR_LR']  # 判别器初始学习率
        weight_decay = self.cfg['WEIGHT_DECAY']  # 权重衰减
        momentum = self.cfg['MOMENTUM']  # 动量

        # 创建判别器的优化器
        discriminator_opt = optim.SGD(self.backbone_discriminator.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

        # 返回优化器字典
        optimizer = {
            'discriminator': discriminator_opt,
        }
        return optimizer


    def generate_fixed_noise(self, images, noise_dim):
        batch_size = images.size(0)
        device = images.device
        noise = torch.zeros(batch_size, noise_dim, device=device)

        for i in range(batch_size):
            image_hash = hashlib.md5(images[i].cpu().numpy().tobytes()).hexdigest()
            seed = int(image_hash, 16) % (2**32)
            torch.manual_seed(seed)
            noise[i] = torch.randn(noise_dim, device=device)

        return noise

    def prepare(self):
        """ common prepare task for training
        """
        for key in self.cfg:
            print(key, self.cfg[key])
        self.make_inputs()
        self.make_model()
        self.backbone_discriminator = models.Discriminator().cuda() # D
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.anta_loss = models.antaGeneratorLoss().cuda()  # 对抗损失
        self.discriminator_loss = models.DiscriminatorLoss().cuda() # D
        self.opt = self.get_optimizer()
        self.opt_discriminator = self.get_discriminator_optimizer() # D
        self.scaler = amp.GradScaler()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'

    def train(self):
        self.prepare()
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone,
                                                device_ids=[self.local_rank], find_unused_parameters=True)
        self.backbone_discriminator = DistributedDataParallel(self.backbone_discriminator,
                                                device_ids=[self.local_rank], find_unused_parameters=True) # D
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")

    def calculate_landmarks(self, inputs):
        size = 112
        inputs = inputs * 0.5 + 0.5  # PFLD requires inputs to be within [0, 1]
        _, landmarks = self.inference_model(inputs)
        landmarks = landmarks.detach().cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
        landmarks = landmarks * [size, size]
        landmark_masks = torch.zeros((landmarks.shape[0], size, size))

        def in_hull(p, hull):
            #  test if points in `p` are in `hull`
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            return hull.find_simplex(p) >= 0

        x, y = np.mgrid[0:size:1, 0:size:1]
        grid = np.vstack((y.flatten(), x.flatten())).T  # swap axes

        for i in range(len(landmarks)):
            hull = ConvexHull(landmarks[i])
            points = landmarks[i, hull.vertices, :]
            mask = torch.from_numpy(in_hull(grid, points).astype(int).reshape(size, size)).unsqueeze(0)
            landmark_masks[i] = mask
        landmark_masks = landmark_masks.unsqueeze(dim=1).cuda()
        landmark_masks.requires_grad = False

        return landmark_masks


def main():
    torch.autograd.set_detect_anomaly(True)
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    # 初始化日志记录器
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=timestamp,  # 替换为你的项目名称
        name="ldd"        # 可选，为当前运行指定名称
    )
    main()

