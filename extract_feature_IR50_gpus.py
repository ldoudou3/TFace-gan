import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from torchkit.backbone import get_model
# 定义图像数据集类
class ImageInferenceDataset(Dataset):
    def __init__(self, datadir, images_name_file_path):
        self.img_paths = self.load_img_paths(datadir, images_name_file_path)
        print(f"Number of images: {len(self.img_paths)}")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def load_img_paths(datadir, images_name_file_path):
        with open(images_name_file_path, 'rb') as f:
            image_names = pickle.load(f)
        print("File names loaded.")
        return [os.path.join(datadir, name) for name in image_names]

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        image = self.transform(image)
        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        return len(self.img_paths)

# 特征提取函数
def extract_features(data_loader, generator):
    embeddings = []
    generator.eval()  # 设置为评估模式

    with torch.no_grad():
        for batch, _ in tqdm(data_loader):
            batch = batch.to(device)
            # 提取特征
            features = generator(batch)
            # 对特征进行 L2 归一化
            x_feature = F.normalize(features, p=2, dim=1)
            embeddings.append(x_feature.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

def main():

    # 加载数据
    dataset = ImageInferenceDataset(data_root, images_name_file_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # 加载生成器模型
    generator = get_model("IR_50")([112, 112])
    generator.load_state_dict(torch.load(generator_ckpt_path), strict=False)
    generator.to(device)

    # 使用 DataParallel 包装模型，以支持多 GPU
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)

    # 提取特征
    print("提取特征中...")
    embeddings = extract_features(data_loader, generator)

    # 保存特征到 .npy 文件
    np.save(output_file, embeddings)
    print(f"特征已保存到 {output_dir}")

if __name__ == "__main__":
    # 配置路径和参数
    generator_ckpt_path = '/home/dhb/dhb3/TFace/tasks/gan/ckpt/Backbone_Epoch_24_checkpoint.pth'  # 替换为生成器的路径
    base_root = '/mnt/nas/dhb/dhb3/CVPR24_FRCSyn_ADMIS/dataset/CASIA/'
    data_root = base_root + 'images'
    images_name_file_path = base_root + 'image_names.pkl'
    output_dir = base_root + 'generator_embeddings'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'generator_embed-16epoch.npy')
    batch_size = 64
    main()