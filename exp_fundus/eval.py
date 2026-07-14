import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io
from skimage import color
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from medpy import metric
from exp_fundus.data.dataset import FundusDataset
from models import networks
from utils import ensure_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/ReMatch', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_FOLDER', help='path to the data')
    parser.add_argument('--train_modality', type=str, default='ORIGA', 
                        choices=['BinRushed', 'Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE'], 
                        help='the image modality for training')
    parser.add_argument('--test_modalities', type=list, default=['BinRushed', 'Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE'], 
                        choices=['BinRushed', 'Drishti_GS', 'Magrabia', 'ORIGA', 'REFUGE'], 
                        help='the image modality for training')
    parser.add_argument('--labeled_percentage', type=float, default=.2, help='the percentage of labeled data')
    parser.add_argument('--model_weights', type=str, default='best.pth', help='model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=3, help='number of target categories')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize predictions')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args, test_modality):
    test_set = FundusDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False, modality=test_modality)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return test_dataloder, len(test_set)


def load_model(model_weights, in_channels, num_classes, backbone='UNet'):
    model = networks.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def eval(is_debug=False):
    args = get_args()
    args.test_modalities.remove(args.train_modality)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}_trainmodal_{}/'.format(args.backbone, args.labeled_percentage, args.train_modality)
    # Load model
    weights_path = project_path + 'weights/' + args.model_weights
    model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    model.eval()
    avg_domain_scores = []
    for test_modality in args.test_modalities:
        # Load Data
        test_dataloader, length = get_data(args=args, test_modality=test_modality)
        iters = len(test_dataloader)
        iter_test_dataloader = iter(test_dataloader)
        
        if is_debug:
            pbar = range(10)
            length = 10 * args.batch_size
        else:
            pbar = range(iters)
        ############################
        # Evaluation
        ############################
        first_total = 0.0
        second_total = 0.0
        print('start evaluation')
        results = {i: [] for i in range(4)}
        with torch.no_grad():
            for idx in tqdm(pbar):
                image, label = next(iter_test_dataloader)
                image, label = image.to(device), label.to(device)
                out = model(image)
                pred = out['out']
                B, C, H, W = label.shape
                pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
                pred = torch.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1).cpu().data.numpy()
                label = label.squeeze(1).long().cpu().numpy()
                
                if np.sum(pred == 1)==0:
                    first_metric = 0,0,0,0
                else:
                    first_metric = calculate_metric_percase(pred == 1, label == 1)

                if np.sum(pred == 2)==0:
                    second_metric = 0,0,0,0
                else:
                    second_metric = calculate_metric_percase(pred == 2, label == 2)
                
                first_total += np.asarray(first_metric)
                second_total += np.asarray(second_metric)
                
                for i in range(4):
                    total_metric = (np.asarray(first_metric) + np.asarray(second_metric)) / 2.
                    results[i].append(total_metric[i])
                
                if args.visualize:
                    image = image.squeeze(0).cpu().numpy() * 255.
                    image = image.transpose([1, 2, 0])
                    label = label[0]
                    label = (color.label2rgb(label, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255.).astype('uint8')
                    pred = pred[0]
                    pred = (color.label2rgb(pred, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255.).astype('uint8')
                    alpha = 0.3
                    label = image.astype(float) / 255 * (1 - alpha) + label.astype(float) / 255 * alpha
                    label = (np.clip(label, 0, 1) * 255.).astype('uint8')
                    pred = image.astype(float) / 255 * (1 - alpha) + pred.astype(float) / 255 * alpha
                    pred = (np.clip(pred, 0, 1) * 255.).astype('uint8')
                    save_path = project_path + 'predictions{}2{}/'.format(args.train_modality, test_modality)
                    ensure_dir(save_path)
                    # io.imsave(save_path + str(idx) + '_img.png', image.astype('uint8'))
                    io.imsave(save_path + str(idx) + '_lbl.png', label)
                    io.imsave(save_path + str(idx) + '_prd.png', pred)
        avg_metric = [first_total / len(pbar), second_total / len(pbar)]
        # save results
        data_frame = pd.DataFrame(
            data={i: results[i] for i in range(4)},
            index=range(1, length + 1))
        data_frame.to_csv(project_path + '/' + 'evaluation_{}2{}.csv'.format(args.train_modality, test_modality), index_label='Index')
        result = data_frame.values
        avg_score = np.mean(result, axis=0)
        avg_std = np.std(result, axis=0, ddof=1)
        with open(project_path+'/performance_{}2{}.txt'.format(args.train_modality, test_modality), 'w') as f:
            f.writelines('class-wise metric is {} \n'.format(avg_metric))
            f.writelines('metric is {} \n'.format(avg_score))
            f.writelines('standard deviation is {}\n'.format(avg_std))
        print('Class AVG Score:{}\n'.format(avg_metric))
        print('AVG Score:{}, Std:{}\n'.format(avg_score, avg_std))
        print(project_path, 'EVAL FINISHED!\n')
        avg_domain_scores.append([avg_score])
    avg_domain_scores = np.concatenate(avg_domain_scores)
    avg_domain_score = np.mean(avg_domain_scores, axis=0)
    with open(project_path+'/performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(avg_domain_score))
    print('AVG Domain Score:{}\n'.format(avg_domain_score))
if __name__ == '__main__':
    eval()
