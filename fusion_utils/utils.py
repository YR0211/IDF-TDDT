import math
import os

import torch

import args


def adjust_learning_rate(optimizer, epoch_count):
    lr = args.args.LR + 0.5 * (args.args.LR_target - args.args.LR) * (
            1 + math.cos((epoch_count - args.args.Warm_epoch) / (args.args.Epoch - args.args.Warm_epoch) * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_learning_rate(optimizer, epoch_count):
    lr = epoch_count * ((args.args.LR_target - args.args.LR) / args.args.Warm_epoch) + args.args.LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def check_dir(base):
    if os.path.isdir(base):
        pass
    else:
        os.makedirs(base)


def rgb2ycbcr(img_rgb):
    b, _, _, _ = img_rgb.shape
    R = torch.unsqueeze(img_rgb[:, 0, :, :], dim=1)
    G = torch.unsqueeze(img_rgb[:, 1, :, :], dim=1)
    B = torch.unsqueeze(img_rgb[:, 2, :, :], dim=1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    img_ycbcr = torch.cat([Y, Cb, Cr], dim=1)
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    b, _, _, _ = img_ycbcr.shape
    Y = torch.unsqueeze(img_ycbcr[:, 0, :, :], dim=1)
    Cb = torch.unsqueeze(img_ycbcr[:, 1, :, :], dim=1)
    Cr = torch.unsqueeze(img_ycbcr[:, 2, :, :], dim=1)
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    img_rgb = torch.cat([R, G, B], dim=1)
    return img_rgb


def to_inference(model, device):
    model.eval()
    model.to(device)
    for param in model.named_parameters():
        param[1].requires_grad = False


def to_train(model, device):
    model.train()
    model.to(device)