import torch
import torch.nn as nn
import os
import sys
import torch.nn.init as init
# from torchvision import transforms
import timm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),  '..'))
# from TinyViT.models.tiny_vit import tiny_vit_5m_224

# 生成器的对抗loss
class antaGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,gen_embed,discriminator):
        # 2. 对抗损失：让判别器将假样本判为真
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
        # self.backbone = tiny_vit_5m_224(pretrained=False) # 输出是 bs * 1000
        # self.backbone.eval()
        
        # 加载 ViT-Small 模型（预训练权重来自 ImageNet-1k）
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        # 获取最后一层特征维度
        self.backbone_features = self.backbone.head.in_features
        # 替换分类头为 Identity，直接获取特征
        self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_features, 512),
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

        # with torch.no_grad():
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