from predict import getMask
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import style.utils as utils
from style.CLIPstyler import getStyleImg
from torchvision import transforms, models
import torch.nn.functional as F
import clip
from PIL import Image

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

prompt_list = [
    {'style':"cloud", 'seed':0},
    {'style':"white wool", 'seed':2},
    {'style':"a sketch with crayon", 'seed':3},
    {'style':"oil painting of flowers", 'seed':7},
    {'style':'pop art of night city', 'seed':6},
    {'style':"Starry Night by Vincent van gogh", 'seed':0},
    {'style':"neon light", 'seed':5},
    {'style':"mosaic", 'seed':4},
    {'style':"green crystal", 'seed':1},
    {'style':"Underwater", 'seed':0},
    {'style':"fire", 'seed':0},
    {'style':'a graffiti style painting', 'seed':2},
    {'style':'The great wave off kanagawa by Hokusai', 'seed':0},
    {'style':'Wheatfield by Vincent van gogh', 'seed':2},
    {'style':'a Photo of white cloud', 'seed':3},
    {'style':'golden', 'seed':2},
    {'style':'Van gogh', 'seed':0},
    {'style':'pop art', 'seed':2},
    {'style':'a monet style underwater', 'seed':3},
    {'style':'A fauvism style painting', 'seed':2}
]

input_data  = [
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/ship.jpg",
        'cris_prompt' : "A white sailboat with three blue sails floating on the sea",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/911.jpg",
        'cris_prompt' : "a plane",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/1.jpg",
        'cris_prompt' : "a flower",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/house.jpg",
        'cris_prompt' : "a house",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/people.jpg",
        'cris_prompt' : "the face",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/Napoleon.jpg",
        'cris_prompt' : "a White War Horse",
        'style_prompts' : prompt_list
    },
    # {
    #     'mask_path' : "./mask/5.jpg",
    #     'img_path' : "./testimg/Napoleon.jpg",
    #     'cris_prompt' : "white horse",
    #     'style_prompts' : prompt_list
    # },
    # {
    #     'mask_path' : "./mask/5.jpg",
    #     'img_path' : "./testimg/Napoleon.jpg",
    #     'cris_prompt' : "horse",
    #     'style_prompts' : prompt_list
    # },
    # ddddddddddddd
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/apple.png",
        'cris_prompt' : "a red apple",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/bigship.png",
        'cris_prompt' : "White Large Luxury Cruise Ship",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/car.png",
        'cris_prompt' : "White sports car.",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/lena.png",
        'cris_prompt' : "A woman's face.",
        'style_prompts' : prompt_list
    },

    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/mountain.png",
        'cris_prompt' : "mountain peak",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/tjl.jpeg",
        'cris_prompt' : "The White House at the Taj Mahal",
        'style_prompts' : prompt_list
    },
    {
        'mask_path' : "./mask/5.jpg",
        'img_path' : "./testimg/man.jpg",
        'cris_prompt' : "The Men's face",
        'style_prompts' : prompt_list
    },

]

# mask_path = "./mask/5.jpg"
# img_path = "./testimg/ship.jpg"
# sent = "the a white with sails on blue"
# sent = "a white boat with blue sails on the sea"
# sent = "A white boat with blue sails gracefully sails across the sea."
# sent = "The boat is a magnificent vessel adorned with three majestic blue sails. Its towering and grand hull commands attention and awe. The pristine white body gleams with a smooth coating, shimmering in the sunlight. Each piece of blue fabric unfurled on the masts exhibits a deep and vibrant shade, creating a striking contrast against the surrounding sea. The sails are securely fastened to the masts with delicate ropes, billowing and dancing in the wind, as if embarking on a journey to the unknown. The beauty and grandeur of this boat are captivating, making it a shining star upon the vast ocean."
# sent = "A white sailboat with three blue sails floating on the sea"

config_path = "./config/refcoco+/test.yaml"
model_pth = "./best_model.pth"

# exit()

def getMaskImg(img,config_path,model_pth,sent=None,isMask=False,):
    if not isMask:
        img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_style2 = img_style1/255.0
        img_style3 = np.transpose(img_style2, (2,0,1))
        img_style4 = torch.Tensor(img_style3)
        img_style = torch.unsqueeze(img_style4, 0)
    
        mask0 = getMask(img,sent,config_path,model_pth)
        mask1 = np.stack((mask0, mask0,mask0), axis=2)
        mask_img = np.array(mask1*255, dtype=np.uint8)
        return mask_img
    else:
        return img 

def getCVImg2Torch(img):
    img_style1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_style1 = img
    img_style2 = img_style1/255.0
    img_style3 = np.transpose(img_style2, (2,0,1))
    img_style4 = torch.Tensor(img_style3)
    img_style = torch.unsqueeze(img_style4, 0)
    return img_style

def load_image(img,mode="PLT"):
    if mode == "CV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(img)[:3, :, :].unsqueeze(0)
    return image.to(device)


# def load2_image(img):
#     # img_cv2 = cv2.imread(img_path)

#     # 将BGR格式转换为RGB格式
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 将OpenCV图像转换为PIL图像
#     image = Image.fromarray(img_rgb)

#     img_width, img_height = image.size
#     if img_width is not None:
#         image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    
#     transform = transforms.Compose([
#                         transforms.ToTensor(),
#                         ])   

#     image = transform(image)[:3, :, :].unsqueeze(0)

#     return image

def squeeze_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)
    image = torch.Tensor(image)
    return image

def img_normalize(image,device):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize2(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    return image

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def getClipFeature(image,clip_model):
    image = F.interpolate(image,size=224,mode='bicubic')
    image = clip_model.encode_image(image.to(device))
    image = image.mean(axis=0, keepdim=True)
    image /= image.norm(dim=-1, keepdim=True)
    return image

def getVggFeature(image,device,VGG):
    return utils.get_features(img_normalize(image,device), VGG)

def getLoss(text_feature,img_feature):
    return 1-torch.cosine_similarity(text_feature, img_feature)

# def getCropImgAndFeature(img,mask,target,clip_model,size=128,batch=64,pot_part=0.8,sizePose=None):
#     back_crop, pot_crop ,pot_aug,extra_pot = [], [],[],[]
#     cropper = transforms.RandomCrop(size)
#     augment = transforms.Compose([
#         transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
#         transforms.Resize(400)
#     ])
#     while len(pot_crop)<batch  :
#         if sizePose:
#             (i, j, h, w) = sizePose
#         else:
#             (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        
#         mask_crop = transforms.functional.crop(mask, i, j, h, w)
#         img_crop = transforms.functional.crop(img, i, j, h, w)
#         target_crop = transforms.functional.crop(target, i, j, h, w)
        
#         if int(mask_crop[0].sum())/(3*size*size) >= pot_part: # 是mask区域 
#             # 这个条件判断裁剪的掩码区域中感兴趣的部分（例如人物）是否占据足够的比例。
#             # 如果掩码区域中的目标像素占总像素的80%或更多，这个区域被认为是重要的，并被选取为潜在的有用裁剪。
#             if len(pot_crop)<batch :
#                 pot_crop.append(img_crop)
#                 pot_aug.append(augment(target_crop))
    
#     pot_allCrop = torch.cat(pot_crop,dim=0)
#     pot_all_crop = pot_allCrop


#     pot_crop_feature = clip_model.encode_image(clip_normalize2(pot_all_crop,device))
#     while len(back_crop) < batch:
#         (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
#         mask_crop = transforms.functional.crop(mask, i, j, h, w)
#         img_crop = transforms.functional.crop(img, i, j, h, w)
#         target_crop = transforms.functional.crop(target, i, j, h, w)
#         if int(mask_crop[0].sum())/(3*size*size) < (1-pot_part):
#             # 这个条件用于识别背景区域。如果掩码区域中目标像素的比例低于10%，
#             # 则认为这个区域主要是背景，并且可能用于提取背景特征或被排除在最终输出之外。
#             img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
#             cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
#             if torch.numel(cos[cos>0.12]) > pot_part*batch:
#                 # 如果大部分裁剪（超过80%）与潜在区域特征的相似度低于0.12，则这些裁剪被认为是背景。
#                 back_crop.append([target_crop,img_crop])
#     while len(extra_pot) < 0.1*batch:
#         (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
#         mask_crop = transforms.functional.crop(mask, i, j, h, w)
#         img_crop = transforms.functional.crop(img, i, j, h, w)
#         target_crop = transforms.functional.crop(target, i, j, h, w)
#         if int(mask_crop[0].sum())/(3*size*size) < (1-pot_part):
#             # 这个条件用于识别背景区域。如果掩码区域中目标像素的比例低于10%，
#             # 则认为这个区域主要是背景，并且可能用于提取背景特征或被排除在最终输出之外。
#             img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
#             cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
#             if torch.numel(cos[cos<0.06]) > (1-pot_part)*batch:
#                 # 如果一定数量的裁剪（超过20%）与潜在区域特征的相似度高于0.06，则这些裁剪被认为足够接近目标区域，可能被用于额外的处理或分析。
#                 extra_pot.append(augment(target_crop))
#                 pot_aug.append(augment(target_crop))
#     return pot_aug,back_crop

# 原始的
def getCropImgAndFeature(img,mask,target,clip_model,size=128,batch=64,pot_part=0.9,sizePose=None):
    print('getCropImgAndFeature begin')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img, mask, target = img.to(device), mask.to(device), target.to(device)
    
    back_crop, pot_crop ,pot_aug,extra_pot = [], [],[],[]
    cropper = transforms.RandomCrop(size)
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(400)
    ])

    max_iterations = 2000  # 设置最大迭代次数以防止无限循环
    iteration = 0

    while len(pot_crop)<batch and iteration < max_iterations :
        iteration += 1
        print(f'iteration={iteration}, len(pot_crop)={len(pot_crop)}')
        print('in while 1: while len(pot_crop)<batch ')
        if sizePose:
            (i, j, h, w) = sizePose
        else:
            (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        
        if int(mask_crop[0].sum())/(3*size*size) >= 0.8:
            if len(pot_crop)<batch :
                print('in while 1: pot_crop.append(img_crop)')
                pot_crop.append(img_crop)
                pot_aug.append(augment(target_crop))
    
    pot_allCrop = torch.cat(pot_crop,dim=0).to(device)
    pot_all_crop = pot_allCrop
    pot_crop_feature = clip_model.encode_image(clip_normalize2(pot_all_crop, device))

    while len(back_crop) < batch and iteration < max_iterations:
        iteration += 1
        print(f'iteration={iteration}, len(back_crop)={len(back_crop)}')
        print('in while 2: len(back_crop) < batch')
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum())/(3*size*size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
            cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            if torch.numel(cos[cos>0.12]) > 0.8*batch:
                print('in while 2: back_crop.append')
                back_crop.append([target_crop,img_crop])
        
    while len(extra_pot) < 0.1*batch and iteration < max_iterations:
        iteration += 1
        print(f'iteration={iteration}, len(extra_pot)={len(extra_pot)}')
        print('in while 3: len(extra_pot) < 0.1*batch')
        (i, j, h, w) = cropper.get_params(squeeze_convert(mask), (size, size))
        # print(f'(i, j, h, w)={(i, j, h, w)}')
        mask_crop = transforms.functional.crop(mask, i, j, h, w)
        img_crop = transforms.functional.crop(img, i, j, h, w)
        target_crop = transforms.functional.crop(target, i, j, h, w)
        if int(mask_crop[0].sum())/(3*size*size) < 0.1:
            img_crop_feature = clip_model.encode_image(clip_normalize2(img_crop,device))
            cos = (1- torch.cosine_similarity(img_crop_feature, pot_crop_feature))
            # if torch.numel(cos[cos<0.06]) > 0.2*batch:
            if torch.numel(cos[cos<0.10]) > 0.1*batch: # plane
                print('in while 3: back_crop.append')
                extra_pot.append(augment(target_crop))
                pot_aug.append(augment(target_crop))

    print('getCropImgAndFeature end')
    return pot_aug, back_crop


import time

def getTotalLoss1(args, content_features,text_features,source_features,text_source,target,device,VGG,clip_model,img,mask):
    print('getTotalLoss1 begin')
    target_features = utils.get_features(img_normalize(target,device), VGG)
    content_loss = 0
    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])

    loss_patch=0 
    # 开始计时
    start_time = time.time()

    # 调用函数
    img_crop, back_crop = getCropImgAndFeature(img, mask, target, clip_model, size=64, batch=64, pot_part=args.pot_part, sizePose=None)

    # 结束计时
    end_time = time.time()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"Function getCropImgAndFeature executed in: {execution_time} seconds")

    img_crop = torch.cat(img_crop,dim=0)
    img_aug = img_crop

    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()
    
    print('compute loss_back')
    loss_back = 0
    lossToBack = torch.nn.MSELoss()
    for i in back_crop:
        a = i[0]
        b = i[1]
        loss_back += lossToBack(a, b)

    # glob_features = clip_model.encode_image(clip_normalize(target,device))
    # glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    # glob_direction = (glob_features-source_features)
    # glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    # loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    loss_glob=0
    
    reg_tv = args.lambda_tv*get_image_prior_losses(target)
    total_loss = args.lambda_patch*loss_patch + args.lambda_c * content_loss+ reg_tv+ args.lambda_dir*loss_glob + args.lambda_c * loss_back

    detail_loss = {
        "loss_patch":loss_patch,
        "content_loss":content_loss,
        "reg_tv":reg_tv,
        "loss_glob":loss_glob,
        "loss_back":loss_back,
    }
    
    print('getTotalLoss1 end')
    return total_loss,detail_loss

def save_image(tensor, filename):
    # 将张量转换为 NumPy 数组
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)
    image = np.transpose(image, (1, 2, 0))

    # 将数据范围从 [0.0, 1.0] 转换为 [0, 255]
    image = image * 255
    image = image.astype(np.uint8)

    # 保存图像
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


import threading
import os

def StyleProcess(mask_path, img_path, cris_prompt, style_prompt, seed, save_epoch=False, size=128, pot_part=0.8):
    tmp_cris = cris_prompt.replace('.','').replace(' ','_')
    tmp_style = style_prompt.replace('.','').replace(' ','_')
    
    base_path = f'/data15/chenjh2309/soulstyler_org/outputs/size={size}/pot_part={pot_part}/{tmp_cris}/seed={seed}_{tmp_style}'
    img_output_image_path = os.path.join(base_path, 'ori_img.png')
    mask_output_image_path = os.path.join(base_path, 'mask_img.png')
    result_output_image_path = os.path.join(base_path, 'result_img.png')
    result_epoch_output_image_path = os.path.join(base_path, 'epoch/')

    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    if save_epoch and not os.path.exists(result_epoch_output_image_path):
        os.makedirs(result_epoch_output_image_path)

    # 已经有结果了
    if os.path.exists(result_output_image_path):
        print(f"File '{result_output_image_path}' already exists. Exiting function.")
        return
    
    img = cv2.imread(img_path)
    # mask_img = cv2.imread(mask_path)
    mask = getMaskImg(img,config_path,model_pth,cris_prompt,isMask=False)

    img = load_image(img,mode="CV")
    mask = load_image(mask)

    # img = load2_image(img)
    # mask = load2_image(mask)

    img = img.to(device)
    mask = mask.to(device)
    # plt.imshow(utils.im_convert2(img))
    # plt.show()
    # plt.imshow(utils.im_convert2(mask))
    # plt.show()

    
    save_image(img, img_output_image_path)
    save_image(mask, mask_output_image_path)

    print("style img start", '='*50)

    output_image = getStyleImg(
        config_path, img, source="a Photo",
        prompt=style_prompt,
        seed=seed,
        get_total_loss=getTotalLoss1,
        mask=mask,
        save_epoch=save_epoch,
        path = result_epoch_output_image_path
    ).to(device)

    # plt.figure(figsize=(20, 20))#6，8分别对应宽和高

    save_image(output_image, result_output_image_path)


def main(case, stylelist):
    global input_data
    for item in input_data[case:case+1]:
        threading_list = []
        print(item)
        # print(item['style_prompts'])
        print(stylelist)
        for sty in item['style_prompts'][stylelist[0]: stylelist[1]]:
            threading_list.append(threading.Thread(target=StyleProcess, 
                                                   args=(item['mask_path'],
                                                         item['img_path'],
                                                         item['cris_prompt'] ,
                                                         sty['style'], 
                                                         sty['seed'], 
                                                         True,
                                                         64, 
                                                         0.95)))

        # 启动子线程
        for t in threading_list:
            t.start()

        # 等待子线程完成
        for t in threading_list:
            t.join()


import argparse

if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--case', type=int, help='An integer for the case')
    parser.add_argument('--stylelist', type=lambda s: [int(item) for item in s.split(',')], help='A list of integers for the style')
    args = parser.parse_args()

    main(args.case, args.stylelist)

# 0-21 Style
# cd soulstyler_org/
# conda activate soulstyler

# ship
# CUDA_VISIBLE_DEVICES=0 python demo.py --case=0 --style=0,7 d02-cuda0
# CUDA_VISIBLE_DEVICES=4 python demo.py --case=0 --style=8,14 d01-cuda4
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=0 --style=15,21 d01-cuda5

# plane
# CUDA_VISIBLE_DEVICES=1 python demo.py --case=1 --style=0,7 # d02-cuda1
# CUDA_VISIBLE_DEVICES=2 python demo.py --case=1 --style=8,14 # d02-cuda2
# CUDA_VISIBLE_DEVICES=3 python demo.py --case=1 --style=15,21 # d02-cuda3

# flower
# CUDA_VISIBLE_DEVICES=5 python demo.py --case=2 --style=0,7 # d02-cuda5
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=2 --style=8,14 # d02-cuda6
# CUDA_VISIBLE_DEVICES=6 python demo.py --case=1 --style=15,21 # d01-cuda6

# CUDA_VISIBLE_DEVICES=4,5,6,7  sh scripts/run.sh /data15/chenjh2309/TeCH/input/cjh/cjh.jpg exp/cjh/

