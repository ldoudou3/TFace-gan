import torch
import torch.nn as nn
import os
import sys
import torch.nn.init as init
# from torchvision import transforms
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),  '..'))
from TinyViT.models.tiny_vit import tiny_vit_5m_224


class RotationModule(nn.Module):
    def __init__(self, feature_dim=512, noise_dim=512):
        super(RotationModule, self).__init__()
        self.rotation_matrix = nn.Parameter(torch.randn(feature_dim + noise_dim, feature_dim))  # 学习的旋转矩阵

    def forward(self, features, noise):
        # 将生成器的输出与噪声拼接在一起
        combined = torch.cat([features, noise], dim=1)  # (batch_size, feature_dim + noise_dim)
        # 应用旋转矩阵
        rotated_features = torch.matmul(combined, self.rotation_matrix)
        return rotated_features

class TransformationNet(nn.Module):
    def __init__(self, input_dim=512):
        super(TransformationNet, self).__init__()

        # 线性层：接受模板特征和噪声，合并后变换
        self.fc1 = nn.Linear(input_dim * 2, 1024)  # 输入模板和噪声拼接后的维度
        self.fc2 = nn.Linear(1024, 512)  # 隐藏层，输出 512 维度的变换特征
        self.fc3 = nn.Linear(512, input_dim)  # 输出变换后的模板特征

    def forward(self, template, noise_embed):
        # 拼接模板特征和噪声向量
        combined_input = torch.cat([template, noise_embed], dim=1)  # [batch_size, 512 * 2]

        # 前向传播通过全连接层
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        transformed_template = self.fc3(x)

        return transformed_template

# 生成器的对抗loss
class antaGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,gen_embed,discriminator):
        # 对抗损失：让判别器将假样本判为真
        criterion = torch.nn.BCEWithLogitsLoss()
        fake_out = discriminator(gen_embed)
        adv_loss = criterion(fake_out, torch.ones_like(fake_out))

        return adv_loss
    
    
# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 输入预处理层，将 512 维特征转换为适合 DINOv2 backbone 的图像输入
        self.pre_backbone = nn.Sequential(
            nn.Linear(512, 3 * 224 * 224),  # 将 512 维扩展到 3x224x224
            nn.ReLU(inplace=False)                # 激活函数
        )
        # 加载 TinyViT-5M 模型
        self.backbone = tiny_vit_5m_224(pretrained=False) # 输出是 bs * 1000
        self.backbone.eval()
        
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),  # 接着一层全连接层
            nn.ReLU(inplace=False),  
            nn.Linear(256, 1)  # 输出一个值：真假判别
            # nn.Sigmoid()
        )

    def forward(self, x):
        # 输入经过预处理层
        processed_input = self.pre_backbone(x)  # (batch_size, 768)
        
        processed_input = processed_input.clone().view(-1, 3, 224, 224)

        with torch.no_grad():
            features = self.backbone(processed_input)
        return self.classifier(features)


# 判别器loss   
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()  # 对抗损失

    def forward(self, real_embed, fake_embed, discriminator,epoch):
        # 在判别器输入中添加高斯噪声，增强生成器的对抗性：
        # self.noise_std = 0.1 * (1 - epoch / 24)  # 随训练进程逐步减小噪声
        # real_embed += torch.randn_like(real_embed) * self.noise_std
        # fake_embed += torch.randn_like(fake_embed) * self.noise_std

        ## 计算真实向量的损失
        real_out = discriminator(real_embed)                        ## 将真实图片放入判别器中
        loss_real_D = self.bce_loss(real_out, torch.ones_like(real_out))  # 真实样本的目标为1

        ## 计算假向量（噪声）的损失
        fake_out = discriminator(fake_embed.detach())                                ## 判别器判断假的图片
        loss_fake_D = self.bce_loss(fake_out, torch.zeros_like(fake_out))  # 假样本的目标为0
              
        loss_D = loss_real_D + loss_fake_D                  ## 损失包括判真损失和判假损失
        # 计算梯度惩罚
        # gradient_penalty = compute_gradient_penalty(discriminator, real_embed, fake_embed)

        return loss_D # +  10 * gradient_penalty


# def compute_gradient_penalty(discriminator, real_data, fake_data):
#     """
#     计算 WGAN 的梯度惩罚项
#     Args:
#         discriminator (nn.Module): 判别器网络
#         real_data (Tensor): 真实数据样本，形状为 (batch_size, feature_dim)
#         fake_data (Tensor): 生成器生成的假样本，形状为 (batch_size, feature_dim)

#     Returns:
#         Tensor: 梯度惩罚值
#     """
#     batch_size = real_data.size(0)
#     alpha = torch.rand(batch_size, 1, device=real_data.device)  # 在 [0, 1] 间采样
#     alpha = alpha.expand_as(real_data)  # 形状扩展到与 real_data 一致

#     # 插值样本
#     interpolates = alpha * real_data + (1 - alpha) * fake_data
#     interpolates.requires_grad_(True)  # 启用梯度计算

#     # 判别器输出
#     d_interpolates = discriminator(interpolates)

#     # 计算梯度
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=torch.ones_like(d_interpolates),
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]

#     # 计算梯度的 L2 范数
#     gradients = gradients.view(batch_size, -1)  # 展平梯度
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # 计算 GP 项

#     return gradient_penalty




def main():
    torch.autograd.set_detect_anomaly(True)
    # 创建判别器
    discriminator = Discriminator()

    # 模拟输入特征 (batch_size, 512)
    input_tensor = torch.randn(4, 512)  # 假设输入特征大小为 512，batch_size 为 4

    # 前向传播
    output = discriminator(input_tensor)

    # 打印输出
    print("Discriminator output shape:", output.shape)  # (batch_size, 1)
    print("Output values:", output)

if __name__ == "__main__":
    main()