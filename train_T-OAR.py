import multiprocessing
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from tqdm import tqdm

import fusion_utils.utils as utils
from args import args as args
from ctdnet.src.load_model import load_ctdnet_model
from ctdnet.src.loss import CTDNet_Loss
from fusion_models import TOAR as oar_model
from fusion_models import base_fusion_network as bfn_model
from fusion_models.dataset import TrainDataset_od, TrainDataset_sod
from fusion_utils.utils import check_dir
from llama import Dialog, Llama
from segformer.tools.load_dataiter import load_dataiter
from segformer.tools.load_model import load_segformer_model
from yolov5.load_model import load_yolov5_model
from yolov5.utils.loss import ComputeLoss

device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:" + device_id if USE_CUDA else "cpu")


def train(epoch, optimizer, t_critertion, bfn, T_dpi, T_net, textf, save_inf, t_info, data_iter):
    T_dpi.train()

    epoch_t_loss = []
    epoch_lbox = []
    epoch_lobj = []
    epoch_lcls = []
    epoch_acc = []
    task_name = None
    if t_info['is_TD']:
        task_name = 'od'
    elif t_info['is_SS']:
        task_name = 'seg'
    elif t_info['is_SOD']:
        task_name = 'sod'

    for step, x in enumerate(data_iter[task_name]):
        if t_info['is_TD']:
            vi = x[0].to(device).bfloat16().detach()  # vis
            ir = x[1].to(device).bfloat16().detach()  # ir
            labels = x[2].to(device).detach()
        elif t_info['is_SS']:
            ir = (x['img'].data[0][:, 0:3, ...] / 255.).to(device).bfloat16().detach()
            vi = (x['img'].data[0][:, 3:, ...] / 255.).to(device).bfloat16().detach()
        elif t_info['is_SOD']:
            ir = x[0].to(device).bfloat16().detach()  # ir
            vi = x[1].to(device).bfloat16().detach()  # vis
            mask = x[2].to(device).bfloat16().detach()
            edge = x[3].to(device).bfloat16().detach()

        vi_ycbcr = utils.rgb2ycbcr(vi)
        vi_Y, vi_Cb, vi_Cr = torch.split(vi_ycbcr, 1, 1)
        ir = ir[:, :1, :, :]

        _, _, height, width = vi.size()

        with torch.no_grad():
            f_img_wofinetune = bfn.forward_ER(ir, vi_Y)

        textf_gen = textf[task_name] + torch.zeros(vi.size(0), textf[task_name].size(1), dtype=textf[task_name].dtype,
                                                   device=textf[task_name].device)

        ir_f1 = bfn.ir_e.forward_obo(ir, 0)
        ir_f1_h = T_dpi.ir_t_dpi.forward_obo(ir_f1, textf_gen, 0)
        ir_f2 = bfn.ir_e.forward_obo(ir_f1_h, 1)
        ir_f2_h = T_dpi.ir_t_dpi.forward_obo(ir_f2, textf_gen, 1)
        ir_f3 = bfn.ir_e.forward_obo(ir_f2_h, 2)
        ir_f3_h = T_dpi.ir_t_dpi.forward_obo(ir_f3, textf_gen, 2)
        ir_f4 = bfn.ir_e.forward_obo(ir_f3_h, 3)

        vi_f1 = bfn.vi_e.forward_obo(vi_Y, 0)
        vi_f1_h = T_dpi.vi_t_dpi.forward_obo(vi_f1, textf_gen, 0)
        vi_f2 = bfn.vi_e.forward_obo(vi_f1_h, 1)
        vi_f2_h = T_dpi.vi_t_dpi.forward_obo(vi_f2, textf_gen, 1)
        vi_f3 = bfn.vi_e.forward_obo(vi_f2_h, 2)
        vi_f3_h = T_dpi.vi_t_dpi.forward_obo(vi_f3, textf_gen, 2)
        vi_f4 = bfn.vi_e.forward_obo(vi_f3_h, 3)

        f_img = bfn.forward_ifvf_ER(ir_f4, vi_f4)

        f_img_c = utils.ycbcr2rgb(torch.cat([f_img, vi_Cb, vi_Cr], dim=1))

        if t_info['is_TD']:
            f_img_c = f_img_c.float()
            preds, train_out = T_net[task_name](f_img_c, augment=False, visualize=False)
            # Compute loss
            t_loss, loss_item = t_critertion[task_name](train_out, labels)
            loss = 2 * t_loss
        if t_info['is_SS']:
            mean = torch.from_numpy(np.array([123.675, 116.28, 103.53])).bfloat16().unsqueeze(dim=0).unsqueeze(
                dim=-1).unsqueeze(dim=-1).to(device)
            std = torch.from_numpy(np.array([58.395, 57.12, 57.375])).bfloat16().unsqueeze(dim=0).unsqueeze(
                dim=-1).unsqueeze(dim=-1).to(device)

            f_img_c1 = f_img_c * 255.
            f_img_c1 = (f_img_c1 - mean) / std

            x['img'].data[0] = f_img_c1

            outputs = T_net[task_name](return_loss=True, **x)
            t_loss = outputs.get('decode.loss_seg')
            acc = outputs.get('decode.acc_seg')
            loss = t_loss

        if t_info['is_SOD']:
            mean = torch.from_numpy(np.array([124.55, 118.90, 102.94])).bfloat16().unsqueeze(dim=0).unsqueeze(
                dim=-1).unsqueeze(dim=-1).to(device)
            std = torch.from_numpy(np.array([56.77, 55.97, 57.50])).bfloat16().unsqueeze(dim=0).unsqueeze(
                dim=-1).unsqueeze(dim=-1).to(device)

            f_img_c1 = f_img_c * 255.
            f_img_c1 = (f_img_c1 - mean) / std

            out1, out_edge, out2, out3, out4, out5 = T_net[task_name](f_img_c1)
            t_loss = t_critertion[task_name](out1, out_edge, out2, out3, out4, out5, mask, edge)
            loss = t_loss

        epoch_t_loss.append(t_loss.item())
        if t_info['is_TD']:
            epoch_lbox.append(loss_item[0].item())
            epoch_lobj.append(loss_item[1].item())
            epoch_lcls.append(loss_item[2].item())
        if t_info['is_SS']:
            epoch_acc.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(T_dpi.parameters(), max_norm=0.5)

        optimizer.step()

        if step % save_inf[task_name + '_save_image_iter'] == 0:
            epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
            if epoch % 10 == 0:
                output_name = save_inf['save_img_dir'] + "/" + epoch_step_name + ".jpg"
                out = f_img_c
                save_image(out, output_name)

                output_name = save_inf['save_img_dir'] + "/" + epoch_step_name + "_wofinetune" + ".jpg"
                out = utils.ycbcr2rgb(torch.cat([f_img_wofinetune, vi_Cb, vi_Cr], dim=1))
                save_image(out, output_name)

        if ((epoch + 1) == args.Epoch and (step + 1) % save_inf[task_name + '_iter_num'] == 0) or (
                epoch % args.save_model_num == 0 and (step + 1) % save_inf[task_name + '_iter_num'] == 0):
            ckpt = {
                "T_dpi": T_dpi.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, os.path.join(save_inf['save_model_dir'], 'ckpt_%s.pth' % (str(epoch))))

    epoch_t_loss_mean = np.mean(epoch_t_loss)
    epoch_lbox_mean = np.mean(epoch_lbox)
    epoch_lobj_mean = np.mean(epoch_lobj)
    epoch_lcls_mean = np.mean(epoch_lcls)
    epoch_acc_mean = np.mean(epoch_acc)

    print(" -epoch " + str(epoch) + " -task " + str(task_name))
    print(" -loss_t_loss " + str(epoch_t_loss_mean))
    print(" -loss_lbox " + str(epoch_lbox_mean))
    print(" -loss_lobj " + str(epoch_lobj_mean))
    print(" -loss_lcls " + str(epoch_lcls_mean))
    print(" -loss_acc " + str(epoch_acc_mean))

    with open(save_inf['save_loss_dir'] + "/" + "loss.txt", 'a') as f:
        f.writelines(" -epoch " + str(epoch) + " -task " + str(task_name) + '\n')
        f.writelines(
            " -loss_t_loss " + str(epoch_t_loss_mean) + '\n')
        f.writelines(
            " -loss_lbox " + str(epoch_lbox_mean) + '\n')
        f.writelines(
            " -loss_lobj " + str(epoch_lobj_mean) + '\n')
        f.writelines(
            " -loss_lcls " + str(epoch_lcls_mean) + '\n')
        f.writelines(
            " -loss_acc " + str(epoch_acc_mean) + '\n')
        f.close()

    if t_info['is_TD']:
        with open(save_inf['save_loss_dir'] + "/" + task_name + "_loss.txt", 'a') as f:
            f.writelines(" -epoch " + str(epoch) + " -task " + str(task_name) + '\n')
            f.writelines(
                " -loss_t_loss " + str(epoch_t_loss_mean) + '\n')
            f.writelines(
                " -loss_lbox " + str(epoch_lbox_mean) + '\n')
            f.writelines(
                " -loss_lobj " + str(epoch_lobj_mean) + '\n')
            f.writelines(
                " -loss_lcls " + str(epoch_lcls_mean) + '\n')
            f.close()
    elif t_info['is_SS']:
        with open(save_inf['save_loss_dir'] + "/" + task_name + "_loss.txt", 'a') as f:
            f.writelines(" -epoch " + str(epoch) + " -task " + str(task_name) + '\n')
            f.writelines(
                " -loss_t_loss " + str(epoch_t_loss_mean) + '\n')
            f.writelines(
                " -loss_acc " + str(epoch_acc_mean) + '\n')
            f.close()
    elif t_info['is_SOD']:
        with open(save_inf['save_loss_dir'] + "/" + task_name + "_loss.txt", 'a') as f:
            f.writelines(" -epoch " + str(epoch) + " -task " + str(task_name) + '\n')
            f.writelines(
                " -loss_t_loss " + str(epoch_t_loss_mean) + '\n')
            f.close()


def main():
    bfn_model_name = "BaseFusionNet"
    oar_model_name = "T_OAR"
    now = int(time.time())
    timeArr = time.localtime(now)
    nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
    save_model_dir = args.train_save_model_dir + "/" + nowTime + "_" + oar_model_name + "_model"
    save_img_dir = args.train_save_img_dir + "/" + nowTime + "_" + oar_model_name + "_img"
    save_loss_dir = save_img_dir + "/loss_map"
    save_model_info_dir = save_img_dir + "/" + "model_info"
    save_feature_dir = save_img_dir + "/" + "Feature"

    save_test_dir = save_img_dir + "/" + "test"
    check_dir(save_test_dir)
    check_dir(save_loss_dir)
    check_dir(save_model_info_dir)
    check_dir(save_model_dir)
    check_dir(save_img_dir)
    check_dir(save_feature_dir)

    # load dataiter
    # object detection
    od_dataset_dir = r'...'

    od_vis_M3FD_train_dir = os.path.join(od_dataset_dir, 'vi', 'train')
    od_ir_M3FD_train_dir = os.path.join(od_dataset_dir, 'ir', 'train')
    od_label_M3FD_train_dir = os.path.join(od_dataset_dir, 'labels', 'train')

    od_vis_dir_list_train = [od_vis_M3FD_train_dir]
    od_ir_dir_list_train = [od_ir_M3FD_train_dir]
    od_label_dir_list_train = [od_label_M3FD_train_dir]
    od_dataset = TrainDataset_od(od_vis_dir_list_train, od_ir_dir_list_train, od_label_dir_list_train,
                                 img_size=[args.t_img_size, args.t_img_size])
    od_data_iter = data.DataLoader(
        dataset=od_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.batch_size,
        collate_fn=TrainDataset_od.collate_fn,
        generator=torch.Generator(device=device)
    )

    # semantic segmentation
    ss_dataset, ss_data_iter = load_dataiter(args.segformer_config, device, args.batch_size)

    # salient object detection
    sod_dataset_dir = r'...'

    sod_vis_dir_list_train = os.path.join(sod_dataset_dir, 'Train', 'RGB')
    sod_ir_dir_list_train = os.path.join(sod_dataset_dir, 'Train', 'T_GRAY')
    sod_label_train_dir = os.path.join(sod_dataset_dir, 'Train', 'GT')
    sod_edge_train_dir = os.path.join(sod_dataset_dir, 'Train', 'Edge')
    sod_dataset = TrainDataset_sod(sod_ir_dir_list_train, sod_vis_dir_list_train, sod_label_train_dir,
                                   sod_edge_train_dir)
    sod_data_iter = data.DataLoader(
        sod_dataset,
        collate_fn=sod_dataset.collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.batch_size,
        generator=torch.Generator(device=device)
    )

    datasets = {
        'od': od_dataset,
        'seg': ss_dataset,
        'sod': sod_dataset
    }

    dataiters = {
        'od': od_data_iter,
        'seg': ss_data_iter,
        'sod': sod_data_iter
    }

    with torch.no_grad():
        llama_generator = Llama.build(
            ckpt_dir=args.llama_ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=512,
            max_batch_size=args.batch_size,
        )

        bfn = getattr(bfn_model, bfn_model_name)
        bfn = bfn(num_blocks=args.num_blocks)
        bfn_ckpt = torch.load(args.bfn_ckpt_dir)
        bfn.load_state_dict(bfn_ckpt['net'])

        t_dpi = getattr(oar_model, oar_model_name)
        T_dpi = t_dpi(num_blocks=args.num_blocks - 1, embed_dim=4096, inC=8)

        # load task network
        # object detection network
        yolov5 = load_yolov5_model(args.yolov5_ckpt_dir, args.yolov5_data_yaml, device)
        # object detection loss
        od_critertion = ComputeLoss(yolov5)

        # semantic segmentation network
        seg = load_segformer_model(args.segformer_config, args.segformer_ckpt_dir, device)

        # salient object detection network
        ctdnet = load_ctdnet_model(mode='train', ckpts_dir=args.ctdnet_ckpt_dir)
        # salient object detection loss
        sod_critertion = CTDNet_Loss()

    for attr_name in dir(llama_generator):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        attr = getattr(llama_generator, attr_name)
        if isinstance(attr, nn.Module):
            attr.eval()
            for param in attr.parameters():
                param.requires_grad = False

    utils.to_inference(bfn, device)
    utils.to_inference(yolov5, device)
    utils.to_inference(seg, device)
    utils.to_inference(ctdnet, device)

    T_nets = {
        'od': yolov5,
        'seg': seg,
        'sod': ctdnet
    }

    T_criterions = {
        'od': od_critertion,
        'sod': sod_critertion
    }

    utils.to_train(T_dpi, device)

    # generate text feature
    with torch.no_grad():
        # object detection
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": "object detection task", },
            ],
        ]
        od_textf = llama_generator.chat_completion_forTQFusion(
            dialogs,
            max_gen_len=None,
        )
        od_textf = torch.mean(od_textf, dim=1)

        # semantic segmentation
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": "semantic segmentation task", },
            ],
        ]
        ss_textf = llama_generator.chat_completion_forTQFusion(
            dialogs,
            max_gen_len=None,
        )
        ss_textf = torch.mean(ss_textf, dim=1)

        # salient object detection
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": "salient object detection task", },
            ],
        ]
        sod_textf = llama_generator.chat_completion_forTQFusion(
            dialogs,
            max_gen_len=None,
        )
        sod_textf = torch.mean(sod_textf, dim=1)

        textfs = {
            'od': od_textf,
            'seg': ss_textf,
            'sod': sod_textf
        }

    del llama_generator
    torch.cuda.empty_cache()

    # optimizer
    optimizer = torch.optim.Adam(T_dpi.parameters(), lr=args.LR, weight_decay=1e-4)

    with open(save_model_info_dir + "/" + "model_info.txt", 'w') as f:
        f.writelines('--------------- start ---------------' + '\n')
        f.writelines('model_name' + '\n')
        f.writelines(oar_model_name)
        for i in T_dpi.children():
            f.writelines(str(i) + '\n')
        f.writelines('---------------- end ----------------' + '\n')
        f.close()

    od_iter_num = int(datasets['od'].__len__() / args.batch_size)
    od_save_image_iter = int(od_iter_num / args.save_image_num)

    ss_iter_num = int(datasets['seg'].__len__() / args.batch_size)
    ss_save_image_iter = int(ss_iter_num / args.save_image_num)

    sod_iter_num = int(datasets['sod'].__len__() / args.batch_size)
    sod_save_image_iter = int(sod_iter_num / args.save_image_num)

    save_inf = {
        'od_save_image_iter': od_save_image_iter,
        'seg_save_image_iter': ss_save_image_iter,
        'sod_save_image_iter': sod_save_image_iter,
        'save_model_dir': save_model_dir,
        'save_loss_dir': save_loss_dir,
        'save_img_dir': save_img_dir,
        "od_iter_num": od_iter_num,
        "seg_iter_num": ss_iter_num,
        "sod_iter_num": sod_iter_num,
    }

    start_epoch = -1
    task_id = 0

    t_info = {
        'is_TD': False,
        'is_SS': False,
        'is_SOD': False,
    }

    for epoch in tqdm(range(start_epoch + 1, args.Epoch)):
        if epoch % args.turn_epoch == 0:
            if task_id == 0:
                t_info = {
                    'is_TD': True,
                    'is_SS': False,
                    'is_SOD': False,
                }
                task_id += 1
            elif task_id == 1:
                t_info = {
                    'is_TD': False,
                    'is_SS': True,
                    'is_SOD': False,
                }
                task_id += 1
            elif task_id == 2:
                t_info = {
                    'is_TD': False,
                    'is_SS': False,
                    'is_SOD': True,
                }
                task_id = 0

        train(epoch, optimizer, T_criterions, bfn, T_dpi, T_nets, textfs, save_inf, t_info, dataiters)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
