from re import template
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import cv2

from torchvision import transforms

def load_image(img,mode="PLT"):
    if mode == "CV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(img)[:3, :, :].unsqueeze(0)
    return image
device = torch.device("cuda:0")
clip_model, _ = clip.load('ViT-B/32', device, jit=False)
clip_model.eval()
txt = "a white ship with fire"
WORD_LENGTH = 77
REAL_LENGTH = 4
img_path = "./testimg/ship.jpg"

img = cv2.imread(img_path)
img = load_image(img,mode="CV")
image = F.interpolate(img,size=224,mode='bicubic')
with torch.no_grad():
    image_features = clip_model.encode_image(image.to(device)).detach()
    # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    tokens = clip.tokenize(txt,context_length=WORD_LENGTH).to(device)
    print(tokens)
    exit()
    # tokens[0,4] = 49407
    # # tokens[0,4] = 0
    # tokens[0,5] = 0
    # tokens[0,6] = 0
    text_features = clip_model.encode_text(tokens)
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features.to(device)
    text_features = text_features.float().unsqueeze(-1).to(device)
    row = tokens[:,1:6].clone()
    
    features = row.float().unsqueeze(-1)

def getFeature(target):
    '''
        tokens : 1 77
        target : 1 5 1
    '''
    target_token = target.squeeze(-1) * row
    
    # target : 1 5
    start = torch.Tensor([[49406]]).to(device)
    end = torch.Tensor([[49407]]).to(device)
    zeros = torch.zeros(1,70).to(device)
    target_token = torch.cat([start,target_token,end,zeros],dim=1)
    #1 512
    text_features = clip_model.encode_text(target_token.clone().type(torch.IntTensor).to(device))
    # text_features.requires_grad_(False)
    #print(text_features.requires_grad)

    return text_features

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

class WordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(REAL_LENGTH, REAL_LENGTH, bias=False)
        self.linear = nn.Linear(REAL_LENGTH, REAL_LENGTH, bias=False)
        self.conv1 = nn.Conv1d(1,64,kernel_size=1)
        self.conv2 = nn.Conv1d(64,256,kernel_size=1)
        self.conv3 = nn.Conv1d(256,256,kernel_size=1)
        self.conv4 = nn.Conv1d(256,1,kernel_size=1)
        self.ln = nn.LayerNorm(REAL_LENGTH)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,token:torch.Tensor):
        token = token[:,1:1+REAL_LENGTH]
        token = self.ln(token)
        # x = self.conv1(token)
        # x1 = self.conv2(x)
        # x2 = self.conv3(x1)
        # x3 = self.conv4(x2)
        x = self.linear(token)
        # x = self.linear(x)
        # x = self.linear(x)

        # x1 = self.linear(token)
        # x1 = self.linear(token)
        # x1 = self.linear(token)

        # x2 = torch.cat([x,x1],dim=0)
        # x3 = self.ln(x2)
        return x

class WordNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 4)
        self.linear2 = nn.Linear(4, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 1)
        self.softmax = nn.Softmax(1)
        

    def forward(self,token:torch.Tensor):
        x = self.linear1(token)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.softmax(x)
        return x

model = WordNet2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
first = None
for epoch in range(300):
    target = model(features).to(device)
    target_feature = getFeature(target.clone())
    loss = torch.cosine_similarity(target*target_feature,target*image_features).mean()
    # print((target*target.T).requires_grad)
    # exit()
    optimizer.zero_grad()
    print(f"{target=}")
    loss.backward()
    optimizer.step()
    scheduler.step()
print("=================")
print(target)
print(features)
exit()

model = WordNet()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# tokens = tokens.unsqueeze(-1)
tokens = tokens.float()
print(tokens.shape)
# exit()
y_pred = model(tokens)

# print(y_pred)
import numpy as np
for epoch in range(20):
    # print("==============")
    y_pred = model(tokens)
    Min = min(y_pred[0])
    Max = max(y_pred[0])
    # print(Min,Max)
    # print(y_pred)
    y_pred4 = y_pred - Min
    y_pred4 = y_pred4 / (Max-Min)
    total = torch.Tensor(np.zeros((1,77))).to(device)
    total[:,1:1+REAL_LENGTH] = y_pred4
    total[:,0] = 1
    total[:,1+REAL_LENGTH] = 1
    # print(y_pred)
    # y_pred2 = F.softmax(y_pred,dim=1)
    tokens2_ = tokens
    tokens2_ = tokens2_.squeeze(-1)
    # tokens2 = tokens2_.masked_fill(y_pred3,0)
    tokens2 = tokens2_* total
    # print(tokens2.long().shape)
    # exit()
    text_features1 = clip_model.encode_text(tokens2.long())
    text_features2 = text_features1.mean(axis=0, keepdim=True)
    text_features3 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
    image_features2 = image_features
    loss = 1-torch.cosine_similarity(text_features3,image_features2)
    # loss = criterion(text_features3,image_features2)
    
    print(epoch, loss.item())
    
    # optimizer.zero_grad()

    loss.backward(retain_graph=True)
print(text_features3.shape)
    # optimizer.step()
# y_pred = model(tokens)
# y_pred2 = F.sigmoid(y_pred)
# print(y_pred2)
# torch.save({"state":model.state_dict()},"best_model.pth")

# print(clip.tokenize("I like a small cat", 17, True).squeeze(0))

