from torch.utils.data import Dataset
from torchvision import transforms
import utils
from PIL import Image
import os
import random
import torch


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pickle
import random


class MatchedImageDataset(Dataset):
    def __init__(self, datadir, images_name_file_path, img_size, mirror=False, random_crop=False, cropping=None):
        """
        初始化数据集，支持按文件名加载，同时保留原有的增强操作。
        
        Args:
            datadir (str): 图像所在的根目录。
            images_name_file_path (str): 文件名列表文件路径（pickle 格式）。
            img_size (int): 图像统一调整的大小。
            mirror (bool): 是否进行水平翻转数据增强。
            random_crop (bool): 是否进行随机裁剪数据增强。
            cropping (callable): 随机裁剪函数（可选）。
        """
        self.datadir = datadir
        self.img_paths,self.labels = self.load_img_paths_and_labels(datadir, images_name_file_path)
        self.num_classes = max(self.labels) + 1  # 计算类别数量
        print(f"Number of images: {len(self.img_paths)}")
        self.mirror = mirror
        self.random_crop = random_crop
        self.cropping = cropping

        # 图像预处理变换
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.aug_image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def load_img_paths(datadir, images_name_file_path):
        # normed templates: (n, 512)
        # "label_dir/name.jpg"
        # 从文件中读取字符串数组  
        with open(images_name_file_path, 'rb') as f:  
            image_names = pickle.load(f)
        print("file names file loaded")
        samples = [] 
        total_images = len(image_names)
        # total_images = min(len(image_names), 10) #在训练时会控制数量，这里就全部提取特征吧

        for index in range(total_images):
            image_path = os.path.join(datadir, image_names[index])
            # print(image_path)
            samples.append(image_path)
        print("image paths file loaded")
        return samples        

    @staticmethod
    def load_img_paths_and_labels(datadir, images_name_file_path):
        """
        Returns:
            list, list: 图像路径列表及其对应的标签。
        """
        with open(images_name_file_path, 'rb') as f:
            image_names = pickle.load(f)
        print("File names file loaded")

        img_paths = []
        labels = []
        for image_name in image_names:
            class_name = os.path.basename(os.path.dirname(image_name))  # 获取子文件夹名作为类别
            class_label = int(class_name)  # 假设子文件夹名是数字，可以直接转换为整数
            # print("class_label", class_label)
            
            img_path = os.path.join(datadir, image_name)

            img_paths.append(img_path)
            labels.append(class_label)

        print("Image paths and labels loaded")
        return img_paths, labels
    
    @staticmethod
    def load_img_paths_and_labels_test(datadir, images_name_file_path):
        """
        Returns:
            list, list: 图像路径列表及其对应的标签。
        """
        with open(images_name_file_path, 'rb') as f:
            image_names = pickle.load(f)
        print("File names file loaded")

        img_paths = []
        labels = []
        for image_name in image_names:
            # class_name = os.path.basename(image_name)  # 获取子文件夹名作为类别
            class_name = os.path.splitext(os.path.basename(image_name))[0]
            class_label = int(class_name)  # 假设子文件夹名是数字，可以直接转换为整数
            # print("class_label", class_label)
            
            img_path = os.path.join(datadir, image_name)

            img_paths.append(img_path)
            labels.append(class_label)

        print("Image paths and labels loaded")
        return img_paths, labels
                                              


    def __getitem__(self, index):
        """
        获取指定索引的图像列表，并进行数据增强。
        
        Args:
            index (int): 数据索引。
        
        Returns:
            tuple: 图像张量和文件名。
        """
        image_path = self.img_paths[index]
        # print("image path:", image_path)
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        

        # 随机裁剪操作
        if self.random_crop and self.cropping:
            image = self.cropping([image])[0]

        # 镜像或普通变换
        if self.mirror and random.random() < 0.5:
            image = self.aug_image_transform(image)
        else:
            image = self.image_transform(image)

        # 返回图像和文件名
        return image, torch.tensor(label,dtype = torch.long)

    def __len__(self):
        """
        返回数据集大小。
        
        Returns:
            int: 数据集样本数量。
        """
        return len(self.img_paths)


# 能用的版本
class MatchedImageDataset_old2(Dataset):
    def __init__(self, datadir, images_name_file_path, img_size, mirror=False, random_crop=False, cropping=None):
        """
        初始化数据集，支持按文件名加载，同时保留原有的增强操作。
        
        Args:
            datadir (str): 图像所在的根目录。
            images_name_file_path (str): 文件名列表文件路径（pickle 格式）。
            img_size (int): 图像统一调整的大小。
            mirror (bool): 是否进行水平翻转数据增强。
            random_crop (bool): 是否进行随机裁剪数据增强。
            cropping (callable): 随机裁剪函数（可选）。
        """
        self.datadir = datadir
        self.img_paths = self.load_img_paths(datadir, images_name_file_path)
        print(f"Number of images: {len(self.img_paths)}")
        self.mirror = mirror
        self.random_crop = random_crop
        self.cropping = cropping

        # 图像预处理变换
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.aug_image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def load_img_paths(datadir, images_name_file_path):
        # normed templates: (n, 512)
        # "label_dir/name.jpg"
        # 从文件中读取字符串数组  
        with open(images_name_file_path, 'rb') as f:  
            image_names = pickle.load(f)
        print("file names file loaded")
        samples = [] 
        total_images = len(image_names)
        # total_images = min(len(image_names), 10) #在训练时会控制数量，这里就全部提取特征吧

        for index in range(total_images):
            image_path = os.path.join(datadir, image_names[index])
            # print(image_path)
            samples.append(image_path)
        print("image paths file loaded")
        return samples                                                      


    def __getitem__(self, index):
        """
        获取指定索引的图像列表，并进行数据增强。
        
        Args:
            index (int): 数据索引。
        
        Returns:
            tuple: 图像张量和文件名。
        """
        image_path = self.img_paths[index]
        # print("image path:", image_path)
        image = Image.open(image_path).convert("RGB")

        # 随机裁剪操作
        if self.random_crop and self.cropping:
            image = self.cropping([image])[0]

        # 镜像或普通变换
        if self.mirror and random.random() < 0.5:
            image = self.aug_image_transform(image)
        else:
            image = self.image_transform(image)

        # 返回图像和文件名
        return image, os.path.basename(image_path)

    def __len__(self):
        """
        返回数据集大小。
        
        Returns:
            int: 数据集样本数量。
        """
        return len(self.img_paths)
    

class MatchedImageDataset_old(Dataset):
    def __init__(self, folders, img_size, mirror=False, random_crop=False, ext=None):
        self.folders = folders
        self.files = utils.list_matching_files(folders, ext=[None, ext])
        self.mirror = mirror
        self.cropping = utils.RandomCrop()
        self.random_crop = random_crop
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.aug_image_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, sample_idx):
        image_list = []

        for i, f in enumerate(self.folders):
            image_list.append(Image.open(os.path.join(f, self.files["dirs"][sample_idx], self.files["files"][sample_idx] + self.files["exts"][sample_idx][i])).convert("RGB"))

        if self.random_crop and random.randint(0, 1) == 0:
            image_list = self.cropping(image_list)  # Perform augmentation through random cropping

        if self.mirror:
            if random.randint(0, 1) == 0:  # Perform horizontal flipping half of the times
                image_list = [self.image_transform(img) for img in image_list]
            else:
                image_list = [self.aug_image_transform(img) for img in image_list]
        else:
            image_list = [self.image_transform(img) for img in image_list]

        return image_list

    def __len__(self):
        return len(self.files["files"])
