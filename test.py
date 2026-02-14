import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier,MedCLIPVisionModel
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip.prompts import generate_class_prompts, process_class_prompts
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams["font.family"] = ["SimHei"]

tokenizer = BertTokenizer.from_pretrained('/home/useryf/MC_ACA/Bio_ClinicalBERT')

def build_image_text_dict_list(mask_dir):
    data_list = []
    train_img_dir = '/home/useryf/MC_ACA/ACA/img/data_train'
    for root, dirs, files in os.walk(train_img_dir):
        subfolder_name = os.path.basename(root)
        count = 0
        for file in files:
            if file.endswith(('.JPG', '.png')):
                img_path = os.path.join(root, file)
                if subfolder_name == 'narrow1':
                    text = 'SL visible, TM visible, SS visible, and CBB partially visible, with a low probability of ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 0
                    })
                elif subfolder_name == 'narrow2':
                    text = 'SL visible, TM visible, SS visible, and CBB invisible, with a low probability of ACA closure'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 1
                    })
                elif subfolder_name == 'narrow3':
                    text = 'SL visible, TM partially visible, SS invisible, and CBB invisible, with a high probability of ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 2
                    })
                elif subfolder_name == 'narrow4':
                    text = 'SL visible, TM invisible, SS invisible, and CBB invisible, ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 3
                    })
                else:
                    text = 'SL visible, TM visible, SS visible, and CBB invisible, ACA open'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 4
                    })
                count += 1
        print(f"{subfolder_name} 子文件夹下加入数据集的图像数量: {count}")
    test_img_dir = '/home/useryf/MC_ACA/ACA/img/data_test'
    for root, dirs, files in os.walk(test_img_dir):
        subfolder_name = os.path.basename(root)
        count = 0
        for file in files:
            if file.endswith(('.JPG', '.png')):
                img_path = os.path.join(root, file)
                if subfolder_name == 'narrow1':
                    text = 'SL visible, TM visible, SS visible, and CBB partially visible, with a low probability of ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 0
                    })
                elif subfolder_name == 'narrow2':
                    text = 'SL visible, TM visible, SS visible, and CBB invisible, with a low probability of ACA closure'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 1
                    })
                elif subfolder_name == 'narrow3':
                    text = 'SL visible, TM partially visible, SS invisible, and CBB invisible, with a high probability of ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 2
                    })
                elif subfolder_name == 'narrow4':
                    text = 'SL visible, TM invisible, SS invisible, and CBB invisible, ACA closure.'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 3
                    })
                else:
                    text = 'SL visible, TM visible, SS visible, and CBB invisible, ACA open'
                    data_list.append({
                        'image': img_path,
                        'text': [text],
                        'label': 4
                    })
                count += 1
        print(f"{subfolder_name} 子文件夹下加入数据集的图像数量: {count}")
    return data_list

class CustomMedDataset(Dataset):
    def __init__(self, data_list, mask_dir, transform=None, prompt_inputs=None):
        self.data_list = data_list
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.prompt_inputs = prompt_inputs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['image']
        text = item['text'][0]
        label = item['label']

        img_filename = os.path.splitext(os.path.basename(img_path))[0]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        encoded_text = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}

        return {
            'pixel_values': image,
            'text': text,
            'input_ids': encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long),
            'prompt_inputs': self.prompt_inputs
        }

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch], dim=0)
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)
    prompt_inputs = batch[0]['prompt_inputs']  # 假设所有样本的prompt_inputs相同
    return {
        'pixel_values': pixel_values,
        'text': [item['text'] for item in batch],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompt_inputs': prompt_inputs
    }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_dir = 'ACA/seg_mask/obj_mask_v3.1'
image_text_dict_list = build_image_text_dict_list(mask_dir)
train_size = int(0.8 * len(image_text_dict_list))
train_data_list, test_data_list = image_text_dict_list[:train_size], image_text_dict_list[train_size:]

cls_prompts = generate_class_prompts(pd.DataFrame({
    'Reports': [
        'SL visible, TM visible, SS visible, and CBB partially visible, with a low probability of ACA closure.',
        'SL visible, TM visible, SS visible, and CBB invisible, with a low probability of ACA closure',
        'SL visible, TM partially visible, SS invisible, and CBB invisible, with a high probability of ACA closure.',
        'SL visible, TM invisible, SS invisible, and CBB invisible, ACA closure.',
        'SL visible, TM visible, SS visible, and CBB invisible, ACA open'
    ],
    'label_0': [1, 0, 0, 0, 0],
    'label_1': [0, 1, 0, 0, 0],
    'label_2': [0, 0, 1, 0, 0],
    'label_3': [0, 0, 0, 1, 0],
    'label_4': [0, 0, 0, 0, 1]
}), task=['label_0', 'label_1', 'label_2', 'label_3', 'label_4'], n=10)
prompt_inputs = process_class_prompts(cls_prompts)

train_dataset = CustomMedDataset(train_data_list, mask_dir, transform=transform, prompt_inputs=prompt_inputs)
test_dataset = CustomMedDataset(test_data_list, mask_dir, transform=transform, prompt_inputs=prompt_inputs)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, drop_last=True)


model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.cuda()


loss_model = ImageTextContrastiveLoss(model)
loss_model.cuda()

train_config = {
    'batch_size':64,
    'num_epochs': 25,
    'warmup': 0.1,  
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 16,
    'eval_steps': 100,
    'save_steps': 100,
}

medclip_clf = PromptClassifier(model)
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=test_dataloader,
    mode='multiclass',
)

train_objectives = [
    (train_dataloader, loss_model, 1),
]

model_save_path = f'./checkpoints/vision_text_pretrain'

trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr': train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    eval_dataloader=test_dataloader,
    use_amp=True,
)

print('训练完成')

model.eval()
pred_list = []
label_list = []
with torch.no_grad():
    for data in test_dataloader:
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        outputs = medclip_clf(**data)
        pred = outputs['logits']
        pred_list.append(pred)
        label_list.append(data['labels'])

pred_list = torch.cat(pred_list, 0)
labels = torch.cat(label_list, 0).cpu().detach().numpy()
pred = pred_list.cpu().detach().numpy()
pred_label = pred.argmax(1)


acc = accuracy_score(labels, pred_label)
sen = recall_score(labels, pred_label, average='macro')
pr = precision_score(labels, pred_label, average='macro')
f1 = f1_score(labels, pred_label, average='macro')

from sklearn.preprocessing import label_binarize
labels_bin = label_binarize(labels, classes=[0, 1, 2, 3, 4])
pred_probs = torch.softmax(torch.tensor(pred), dim=1).numpy()
auc = roc_auc_score(labels_bin, pred_probs, average='macro')

print(f"ACC: {acc:.4f}")
print(f"SEN: {sen:.4f}")
print(f"PR: {pr:.4f}")
print(f"AUC: {auc:.4f}")

print(f"F1: {f1:.4f}")
