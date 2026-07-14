import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from exp_mscmr.data.dataset import MSCMRDataset
from models import networks
from wheels.loss_functions import DSCLossH
from wheels.logger import logger as logging
from utils import ensure_dir, instance_max_min_normalize
from wheels.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from wheels.torch_utils import seed_torch
from wheels.model_init import init_weight
from wheels.activation_extraction import ActivationExtraction
from wheels.rom_utils import ROCTransform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/ReMatch', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_FOLDER', help='path to the data')
    parser.add_argument('--train_modality', type=str, default='C0', choices=['C0', 'LGE'], help='the image modality for training')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=.1, help='the percentage of labeled data')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=1, help='input channels')
    parser.add_argument('--num_classes', type=int, default=4, help='number of target categories')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--log_freq', type=float, default=10, help='logging frequency of metrics accord to the current iteration')
    parser.add_argument('--save_freq', type=float, default=10, help='saving frequency of model weights accord to the current epoch')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    val_set = MSCMRDataset(image_path=args.data_path, stage='val', image_size=args.image_size, is_augmentation=False, modality=args.train_modality)
    labeled_train_set = MSCMRDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage, modality=args.train_modality)
    unlabeled_train_set = MSCMRDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=False, percentage=args.labeled_percentage, modality=args.train_modality)
    train_set = ConcatDataset([labeled_train_set, unlabeled_train_set])

    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    print('before: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    print('after: ', len(train_set), len(labeled_train_set), len(unlabeled_train_set), len(val_set))
    assert len(labeled_train_set) == len(train_set)
    train_labeled_dataloader = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    aux_unlabeled_dataloader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    return train_labeled_dataloader, train_unlabeled_dataloader, val_dataloader, aux_unlabeled_dataloader


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}_trainmodal_{}/'.format(args.backbone, args.labeled_percentage, args.train_modality)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_s_loss = []
    metrics.train_d_loss = []
    metrics.train_u_loss = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    train_labeled_dataloader, train_unlabeled_dataloader, val_dataloader, aux_unlabeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)
    val_iters = len(val_dataloader)

    # Load Model
    student = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    denoise_model = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=1).to(device)
    init_weight(denoise_model.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    activation_extractor = ActivationExtraction(model=student, texture_layer=student.net.backbone.in_conv, shape_layer=student.net.classifier.up4).to(device)
    best_model_wts = copy.deepcopy(student.state_dict())
    logger.info('#parameters: {}'.format(sum(param.numel() for param in student.parameters())))
    best_epoch = 0
    best_loss = 100
    conf_threshold = 0.95  # the confidence threshold for pixel-level pseudo label selection

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLossH(num_classes=args.num_classes, device=device)
    criterion_u = DSCLossH(num_classes=args.num_classes, device=device)
    criterion_d = nn.MSELoss()
    optimizer = optim.AdamW([{'params':student.parameters()}, 
                             {'params':denoise_model.parameters()}], lr=args.learning_rate, betas=(0.9, 0.999))
    roc_transformation_u = ROCTransform(p=.5)
    roc_transformation_l = ROCTransform(p=.5)

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_s_loss = []
        epoch_metrics.train_d_loss = []
        epoch_metrics.train_u_loss = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)
        iter_aux_unlabeled_loader = iter(aux_unlabeled_dataloader)

        ############################
        # Train
        ############################
        student.train()
        denoise_model.train()
        for idx in pbar:
            # sparse labeled data
            limage, label, limageA1, limageA2 = next(iter_train_labeled_dataloader)
            limage, label = limage.to(device), label.to(device)
            limageA1, limageA2 = limageA1.to(device), limageA2.to(device)
            # unlabel data
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage, uimageA1, uimageA2 = uimage.to(device), uimageA1.to(device), uimageA2.to(device)
            # auxiliary data
            uaimage, _, uaimageA1, uaimageA2, uamask = next(iter_aux_unlabeled_loader)
            uaimage = uaimage.to(device)
            uaimageA1, uaimageA2, uamask = uaimageA1.to(device), uaimageA2.to(device), uamask.to(device).long()

            optimizer.zero_grad()

            # pseudo labelling
            with torch.no_grad():
                pred_uo = student(uimage)
                pred_uo_logits = pred_uo['out']
                pred_uo_probs = torch.softmax(pred_uo_logits, dim=1) 
                pred_uo_pseudo = torch.argmax(pred_uo_probs, dim=1).detach() 
                pred_uo_conf = pred_uo_probs.max(dim=1)[0].clone()
            # the forward pass of unlabeled images    
            for j in range(uimageA1.shape[0]):
                uimageA1[j] = roc_transformation_u(uimageA1[j], uaimageA1[j])
            pred_uA1 = activation_extractor(uimageA1)
            pred_uA1_logits = pred_uA1['out']
            # unlabeled image synthesis
            with torch.no_grad():
                texture_map_u = activation_extractor.texture_activation[0]
                shape_map_u = activation_extractor.shape_activation[0]
                texture_map_u = torch.mean(texture_map_u, dim=1, keepdim=True)
                shape_map_u = torch.mean(shape_map_u, dim=1, keepdim=True)
                Utu, Stu, Vhtu = torch.linalg.svd(texture_map_u, full_matrices=False)   # B C H H; B C H; B C W W;
                Usu, Ssu, Vhsu = torch.linalg.svd(shape_map_u, full_matrices=False)
                wu = torch.rand(Stu.shape[-1]).view(1, 1, -1).to(device)
                Su = Stu * wu + Ssu * (1 - wu)
                uimageA2 = Usu @ torch.diag_embed(Su) @ Vhsu
                uimageA2 = torch.clamp(uimageA2, 0, 1)
                denoise_model.eval()
                uimageA2 = denoise_model(uimageA2)['out'].detach()
                
                p5_uimageA2 = torch.quantile(uimageA2, 0.01)
                p95_uimageA2 = torch.quantile(uimageA2, 0.99)
                uimageA2 = torch.clamp(uimageA2, p5_uimageA2, p95_uimageA2)
                uimageA2 = instance_max_min_normalize(uimageA2)
                uimageA2[uimageA1 == 0] = 0
                
            denoise_model.train()
            # the forward pass of synthesized unlabeled images
            pred_uA2 = student(uimageA2)
            pred_uA2_logits = pred_uA2['out']
            
            # the forward pass of labeled images
            for i in range(limageA2.shape[0]):
                limageA1[i] = roc_transformation_l(limageA1[i], uaimageA2[i])
            pred_lA1 = activation_extractor(limageA1)
            pred_lA1_logits = pred_lA1['out']
            with torch.no_grad():
                texture_map_l = activation_extractor.texture_activation[0]
                shape_map_l = activation_extractor.shape_activation[0]
                texture_map_l = torch.mean(texture_map_l, dim=1, keepdim=True)
                shape_map_l = torch.mean(shape_map_l, dim=1, keepdim=True)
                Utl, Stl, Vhtl = torch.linalg.svd(texture_map_l, full_matrices=False)   # B C H H; B C H; B C W W;
                Usl, Ssl, Vhsl = torch.linalg.svd(shape_map_l, full_matrices=False)
                wl = torch.rand(Stl.shape[-1]).view(1, 1, -1).to(device)
                Sl = Stl * wl + Ssl * (1 - wl)
                limageA2 = Usl @ torch.diag_embed(Sl) @ Vhsl
                limageA2 = torch.clamp(limageA2, 0, 1).detach()
            limageA2 = denoise_model(limageA2)['out']
            limageA2 = instance_max_min_normalize(limageA2)
            
            # the supervised loss
            loss_s = criterion(pred_lA1_logits, label.squeeze(1).long()) 
            loss_d = criterion_d(limageA2, limageA1.detach()) * 10.
            # the unsupervised loss
            loss_u = (criterion_u(pred_uA1_logits, pred_uo_pseudo, pixel_mask=(pred_uo_conf >= conf_threshold).float()) \
                + criterion_u(pred_uA2_logits, pred_uo_pseudo, pixel_mask=(pred_uo_conf >= conf_threshold).float())) / 2.

            loss = (loss_s + loss_d + loss_u) / 3.

            loss.backward()
            optimizer.step()

            include_rate = (pred_uo_conf >= conf_threshold).sum() / pred_uo_conf.numel()
            writer.add_scalar('train_loss_s', loss_s.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss_d', loss_d.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss_u', loss_u.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('include_rate', include_rate, idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train: Epoch/Epochs {}/{}, iter/iters {}/{}, "
                            "loss {:.3f}, loss_s {:.3f}, loss_d {:.3f}, loss_u {:.3f}, included_rate {:.3f}"
                            .format(epoch, args.num_epochs, idx, len(pbar), 
                                    loss.item(), loss_s.item(), loss_d.item(), loss_u.item(), include_rate))
            epoch_metrics.train_loss.append(loss.item())
            epoch_metrics.train_s_loss.append(loss_s.item())
            epoch_metrics.train_d_loss.append(loss_d.item())
            epoch_metrics.train_u_loss.append(loss_u.item())
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_s_loss.append(np.mean(epoch_metrics.train_s_loss))
        metrics.train_d_loss.append(np.mean(epoch_metrics.train_d_loss))
        metrics.train_u_loss.append(np.mean(epoch_metrics.train_u_loss))

        ############################
        # Validation
        ############################
        epoch_metrics.val_loss = []
        iter_val_dataloader = iter(val_dataloader)
        if is_debug:
            val_pbar = range(10)
        else:
            val_pbar = range(val_iters)
        student.eval()
        with torch.no_grad():
            for idx in val_pbar:
                image, label = iter_val_dataloader.next()
                image, label = image.to(device), label.to(device)
                pred = student(image)['out']
                loss = criterion(pred, label.squeeze(1).long())
                writer.add_scalar('val_loss', loss.item(), idx + len(val_pbar) * (epoch-1))
                if idx % args.log_freq == 0:
                    logger.info("Val: Epoch/Epochs {}/{}\t"
                                "iter/iters {}/{}\t"
                                "loss {:.3f}".format(epoch, args.num_epochs, idx, len(val_pbar),
                                                     loss.item()))
                epoch_metrics.val_loss.append(loss.item())
        metrics.val_loss.append(np.mean(epoch_metrics.val_loss))

        # Save Model
        if np.mean(epoch_metrics.val_loss) <= best_loss:
            best_model_wts = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_loss = np.mean(epoch_metrics.val_loss)
            torch.save(best_model_wts, save_path + 'best.pth'.format(best_epoch))
        torch.save(student.state_dict(), save_path + 'last.pth'.format(best_epoch))
        logger.info("Average: Epoch/Epoches {}/{}, "
                    "train epoch loss {:.3f}, "
                    "val epoch loss {:.3f}, "
                    "best loss {:.3f} at {}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),
                                                     np.mean(epoch_metrics.val_loss), best_loss, best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'loss_s': metrics.train_s_loss,
              'loss_u': metrics.train_u_loss,
              'val_loss': metrics.val_loss},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_val_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.val_loss, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_val_loss.png')

    print(project_path)
    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()

