#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import os
import argparse
from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm 

from models.model_dict import get_model
from Mydataset import ISBICellDataloader, Drosophila_Dataloader
import utils as ut


####### dataset loader ########
def data_loader(args):
    if args.modelname=='SAMUS':
        test_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])
                          
                                   
    else:
        test_transform = ut.ExtCompose([ut.ExtResize((1024, 1024)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    if args.dataset=='ISBI2012':
        data_test = ISBICellDataloader(root = args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
        
    elif args.dataset=='ssTEM':
        data_test = Drosophila_Dataloader(rootdir=args.datapath+args.dataset, val_area=1, split='test', transform=test_transform)
           
           
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=False, drop_last=True, num_workers=2)


    return test_loader



####### IoU ##########
def fast_hist(label_true, label_pred, classes):
    mask = (label_true >= 0) & (label_true < classes)
    hist = np.bincount(classes * label_true[mask].astype(int) + label_pred[mask], minlength=classes ** 2,).reshape(classes, classes)
    return hist


def IoU(output, target, label):
    output = torch.stack(output)
    target = torch.stack(target)

    if label==2:
        output = torch.where(torch.sigmoid(output)>=0.5, 1, 0)
        seg = np.array(output[:,0])
        target = np.array(target[:,0])
    else:
        output = F.softmax(output, dim=1)
        _, output_idx = output.max(dim=1)
        seg = np.array(output_idx)
        target = np.array(target)
    

    # onehot
    confusion_matrix = np.zeros((label, label))

    for lt, lp in zip(target, seg):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten(), label)

    diag = np.diag(confusion_matrix)
    iou_den = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - (diag+1e-7))
    iou = (diag+1e-7) / np.array(iou_den, dtype=np.float32)
    return iou

####### Validation #######
def test():
    model_path = "{}/model/model_bestiou.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            
            if args.num_classes==2:
                targets = targets.unsqueeze(1)
                targets = targets.float()
            else:
                targets = targets.long() 
                
            ##### model input #####
            output = model(inputs, None, None)
            output = output['masks']
            
            
            ##### IoU ######
            output = output.cpu()
            targets = targets.cpu()
            for j in range(output.shape[0]):
                predict.append(output[j])
                answer.append(targets[j])
                
        iou = IoU(predict, answer, label=args.num_classes)


    return iou



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment Anything Model')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num_epochs',  type=int, default=200)
    parser.add_argument('--dataset',  type=str, default='ISBI2012', help='ISBI2012 or ssTEM')
    parser.add_argument('--datapath',  type=str, default='./Dataset/')
    parser.add_argument('--num_classes',  type=int, default=2)
    parser.add_argument('--multimask_output', type=bool, default=False)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--modelname', default='SAM', type=str, help='SAM, MobileSAM, SAM_LoRA, SAM_ConvLoRA, SAM_AdaptFormer, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='./models/Pretrained_model/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()
    
    
    ##### GPU #####
    gpu_flag = args.gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
        
        
    ########## SAM model ##########
    if 'Mobile' in args.modelname:
        args.sam_ckpt = 'models/Pretrained_model/mobile_sam.pt'
        
    model = get_model(args.modelname, args=args).to(device)

    if args.multi:
        model = nn.DataParallel(model)
        
        
    ######### Dataset ##########
    test_loader = data_loader(args)
    
    
    mm = test()
    

    print("  mIoU   : %.2f" % (np.mean(mm)*100.))
    



