#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('nvidia-smi')


# # Library

# In[ ]:


# !pip install transformers[ja]
# !pip install --quiet sentencepiece


# In[60]:


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import random
import gc

import functools
print = functools.partial(print, flush=True)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import confusion_matrix
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoTokenizer, AutoModel, MMBTForClassification, MMBTConfig, AutoConfig,
    Trainer, TrainingArguments, T5Tokenizer
)
import transformers

from torchvision.io import read_image
from torchvision.models import ResNet152_Weights, resnet152, ViT_B_16_Weights, vit_b_16, EfficientNet_V2_M_Weights, efficientnet_v2_m

from matplotlib import pyplot as plt
import seaborn as sns


# In[61]:


print(transformers.__version__)


# In[77]:


class CFG:
    vision_model_weight=EfficientNet_V2_M_Weights.IMAGENET1K_V1
    text_model="cl-tohoku/bert-base-japanese-v2"
    max_len=48
    seed=42
    n_fold=7
    num_labels=2
    num_train_epochs=3
    learning_rate=3e-5
    batch_size=16
    num_workers=0
    target_cols=["is_laugh"]
    output_dir="./models/"
    fp16=False
    debug=False


# # Setting

# In[63]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(42)


# In[64]:


INPUT = "../bokete" # 所望のディレクトリに変更してください。
train_image_path = "../bokete/train/"
test_image_path = "../bokete/test/"

train_df = pd.read_csv(os.path.join(INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT, "test.csv"))
submission_df = pd.read_csv(os.path.join(INPUT, "sample_submission.csv"))

train_df["img_path"] = train_image_path + train_df["odai_photo_file_name"]
test_df["img_path"] = test_image_path + test_df["odai_photo_file_name"]


# In[65]:


print(f"train_data: {train_df.shape}")
print(train_df.head())

print(f"test_data: {test_df.shape}")
print(test_df.head())


# In[66]:


Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[CFG.target_cols])):
    train_df.loc[val_index, 'fold'] = int(n)
train_df['fold'] = train_df['fold'].astype(int)
print(train_df.groupby('fold').size())


# In[67]:


print(train_df[train_df["fold"] == 0]["is_laugh"].value_counts())


# In[68]:


test_df["is_laugh"] = 0


# # MMBT
# MMBTとはMultiModal BiTransformersの略であり、BERTをベースとした画像とテキストのマルチモーダルディープラーニングです。画像にはResNet152を、テキスト側はBERTを用いてそれぞれベクトル変換し、両方をtokenとして連結したものに再度BERTに入力します。  
# https://arxiv.org/pdf/1909.02950.pdf
# 
# https://github.com/facebookresearch/mmbt
# 
# すでにhuggingface内にモデルがあるので、今回はこちらを使用していきたいと思います。
# https://huggingface.co/docs/transformers/main/en/model_summary#multimodal-models
# 

# In[69]:


# 画像データをEmbeddingしていきます
# class ImageEncoder(nn.Module):
#     POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}
#     def __init__(self, pretrained_weight):
#         super().__init__()
#         model = resnet152(weights=pretrained_weight)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.pool = nn.AdaptiveAvgPool2d(self.POOLING_BREAKDOWN[3])

#     def forward(self,  x):
#         out = self.pool(self.model(x))
#         out = torch.flatten(out, start_dim=2)
#         out = out.transpose(1, 2).contiguous()
#         return out

class ImageEncoder(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        self.model = efficientnet_v2_m(weights=pretrained_weight)
        self.fc = nn.Linear(self.model.classifier[1].in_features, 2048)
        
    def _forward_impl(self, x):
        x = self.model.features(x)
        # print(x.size())
        x = torch.flatten(x, 2)
        # x = self.model.avgpool(x)
        # print(x.size())
        x = x.permute(0, 2, 1)
        # print(x.size())
        
        x = self.fc(x)
        # print(x.size())
        return x

    def forward(self, x):
        return self._forward_impl(x)



# In[70]:


def read_jpg(path):
    image_tensor = read_image(path)
    if image_tensor.shape[0] == 1:
        # 1channel=白黒画像があるので3channelにconvertしています。
        image_tensor = image_tensor.expand(3, *image_tensor.shape[1:])
    return image_tensor

class BoketeTextImageDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_len:int, image_transform):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_transforms = image_transform.transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sentence = torch.tensor(self.tokenizer.encode(row["text"], max_length=self.max_seq_len, padding="max_length", truncation=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_len]

        image = self.image_transforms(read_jpg(row["img_path"]))

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": torch.tensor(row["is_laugh"]),
        }

def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return {
        "input_ids":text_tensor,
        "attention_mask":mask_tensor,
        "input_modal":img_tensor,
        "modal_start_tokens":img_start_token,
        "modal_end_tokens":img_end_token,
        "labels":tgt_tensor,
    }


# 学習済みモデルには、東北大学の乾研究室が作成したものを使用します。

# In[71]:


tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
# tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
# tokenizer.do_lower_case = True


# # Data Split

# In[74]:


def train_loop(folds, fold):
    
    print(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    if CFG.debug:
        train_folds = train_folds[:100]
        
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    trn_ds = BoketeTextImageDataset(train_folds, tokenizer, CFG.max_len, 
                                    image_transform=CFG.vision_model_weight)
    val_ds = BoketeTextImageDataset(valid_folds, tokenizer, CFG.max_len, 
                                    image_transform=CFG.vision_model_weight)
    
    transformer_config = AutoConfig.from_pretrained(CFG.text_model)
    transformer = AutoModel.from_pretrained(CFG.text_model)
    
    config = MMBTConfig(transformer_config, num_labels=CFG.num_labels)
    model = MMBTForClassification(config, transformer,
                                  ImageEncoder(CFG.vision_model_weight))
    
    config.use_return_dict = True
    model.config = model.mmbt.config
    
    output_dir = os.path.join(CFG.output_dir, f"fold{str(fold)}")

    trainer_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=CFG.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        learning_rate=CFG.learning_rate,
        save_total_limit=1,
        fp16=CFG.fp16,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        logging_dir='./logs',
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=trn_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    # trainer.train(resume_from_checkpoint=None)
    trainer.train()
    
    val_preds = trainer.predict(val_ds).predictions
    
    valid_folds[[f"pred_{i}" for i in range(CFG.num_labels)]] = val_preds

    # sanity check
    score = log_loss(valid_labels, softmax(val_preds, axis=-1))
    print(f"Score: {score:.4f}")
    
    acc_score = accuracy_score(valid_labels, np.argmax(val_preds, axis=-1))
    print(f"Acc Score: {acc_score:.4f}")
    
    torch.save({'model': model.state_dict(),
                'predictions': val_preds},
                os.path.join(output_dir, f"fold{str(fold)}_best.pth"))
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds


# In[75]:


def get_result(oof_df):
    labels = oof_df[CFG.target_cols].values
    val_preds = oof_df[[f"pred_{i}" for i in range(CFG.num_labels)]].values
    score = log_loss(oof_df["is_laugh"].values, softmax(val_preds, axis=-1))
    print(f'Score CV: {score:<.4f}')
    
        
oof_df = pd.DataFrame()
for fold in range(CFG.n_fold):
    _oof_df = train_loop(train_df, fold)
    oof_df = pd.concat([oof_df, _oof_df])
    print(f"========== fold: {fold} result ==========")
    get_result(_oof_df)
    
oof_df = oof_df.reset_index(drop=True)
print(f"========== CV ==========")
get_result(oof_df)
oof_df.to_csv('oof_df.csv', index=False)    


# # Predict

# In[91]:


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(**inputs)
            y_preds = y_preds.logits
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    predictions = softmax(predictions, axis=-1)
    return predictions


# In[92]:


test_ds = BoketeTextImageDataset(test_df, tokenizer, CFG.max_len,
                                 image_transform=CFG.vision_model_weight)
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
    drop_last=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predictions = []
for fold in range(CFG.n_fold):
    transformer_config = AutoConfig.from_pretrained(CFG.text_model)
    transformer = AutoModel.from_pretrained(CFG.text_model)
    
    config = MMBTConfig(transformer_config, num_labels=CFG.num_labels)
    model = MMBTForClassification(config, transformer,
                                  ImageEncoder(CFG.vision_model_weight))
    model.to(device)
    
    config.use_return_dict = True
    model.config = model.mmbt.config
    
    output_dir = os.path.join(CFG.output_dir, f"fold{str(fold)}")
    state = torch.load(os.path.join(output_dir, f"fold{str(fold)}_best.pth"),
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
    
predictions = np.mean(predictions, axis=0)


# In[93]:


submission_df["is_laugh"] = predictions[:, 1]


# In[94]:


submission_df["is_laugh"] = submission_df["is_laugh"].astype(float)


# In[95]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:




