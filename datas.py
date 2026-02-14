import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams["font.family"] = ["SimHei"]

def build_image_text_dict_list(mask_dir):
    data_list = []
    def check_mask_exists(img_path, text):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = img_filename + '.png'
        mask_folder = os.path.join(mask_dir, text)
        mask_path = os.path.join(mask_folder, mask_filename)
        return os.path.exists(mask_path)

    train_img_dir = 'ACA/img/data_train'
    for root, dirs, files in os.walk(train_img_dir):
        subfolder_name = os.path.basename(root)
        for file in files:
            if file.endswith(('.JPG', '.png')):
                img_path = os.path.join(root, file)
                if check_mask_exists(img_path, subfolder_name):
                    data_list.append({
                        'image': img_path,
                        'text': [subfolder_name]
                    })

    test_img_dir = 'ACA/img/data_test'
    for root, dirs, files in os.walk(test_img_dir):
        subfolder_name = os.path.basename(root)
        for file in files:
            if file.endswith(('.JPG', '.png')):
                img_path = os.path.join(root, file)
                if check_mask_exists(img_path, subfolder_name):
                    data_list.append({
                        'image': img_path,
                        'text': [subfolder_name]
                    })
    return data_list

class CustomMedDataset(Dataset):
    def __init__(self, data_list, mask_dir, transform=None):
        self.data_list = data_list
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['image']
        text = item['text'][0]

        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_folder = os.path.join(self.mask_dir, text)
        mask_path = os.path.join(mask_folder, f"{img_filename}.png")


        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')


        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, text

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_dir = 'ACA/seg_mask/obj_mask_v3.1'
image_text_dict_list = build_image_text_dict_list(mask_dir)
train_size = int(0.8 * len(image_text_dict_list))
train_data_list, test_data_list = image_text_dict_list[:train_size], image_text_dict_list[train_size:]

train_dataset = CustomMedDataset(train_data_list, mask_dir, transform=transform)
test_dataset = CustomMedDataset(test_data_list, mask_dir, transform=transform)

def visualize_samples(dataset, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        image, mask, text = dataset[i]

        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        mask_np = mask.squeeze().numpy()

        # 显示原始图片
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'图片 - {text}', fontsize=12)
        axes[i, 0].axis('off')

        # 显示掩码（灰度图）
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('掩码', fontsize=12)
        axes[i, 1].axis('off')

        # 显示叠加效果
        overlay = img_np.copy()
        overlay[mask_np > 0.5] = [1, 0, 0]  
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('掩码叠加', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
visualize_samples(train_dataset)
visualize_samples(test_dataset)