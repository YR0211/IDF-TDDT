import multiprocessing
import os
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

import fusion_utils.utils as utils
from args import args as args
from fusion_models import TOAR as dpi_model
from fusion_models import base_fusion_network as bfn_model
from fusion_models.dataset import TestDataset
from fusion_utils.utils import check_dir
from llama import Dialog, Llama

device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:" + device_id if USE_CUDA else "cpu")


def main():
    bfn_name = 'BaseFusionNet'
    dpi_name = 'T_OAR'
    now = int(time.time())
    timeArr = time.localtime(now)
    nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
    save_img_dir = './results/' + nowTime + '_' + bfn_name + '_' + dpi_name
    save_compare_dir = save_img_dir + "/compare"
    save_eq_fusion_dir = save_img_dir + "/eq_fusion"
    check_dir(save_img_dir)
    check_dir(save_img_dir)
    check_dir(save_eq_fusion_dir)
    check_dir(save_compare_dir)

    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()  # (0, 255) -> (0, 1)
    ])

    dataset_dir = r'...'

    vis_test_dir = os.path.join(dataset_dir, 'vi', 'test')
    ir_test_dir = os.path.join(dataset_dir, 'ir', 'test')

    with torch.no_grad():
        llama_generator = Llama.build(
            ckpt_dir=args.llama_ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=512,
            max_batch_size=args.batch_size,
        )

        bfn = getattr(bfn_model, bfn_name)
        bfn = bfn(num_blocks=args.num_blocks)
        bfn_ckpt = torch.load(args.bfn_ckpt_dir)
        bfn.load_state_dict(bfn_ckpt['net'])

        t_dpi = getattr(dpi_model, dpi_name)
        T_dpi = t_dpi(num_blocks=args.num_blocks - 1, embed_dim=4096, inC=8)

    for attr_name in dir(llama_generator):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        attr = getattr(llama_generator, attr_name)
        if isinstance(attr, nn.Module):
            attr.eval()
            for param in attr.parameters():
                param.requires_grad = False

    utils.to_inference(bfn, device)
    utils.to_inference(T_dpi, device)

    bfn_checkpoint = torch.load(r'...')
    bfn.load_state_dict(bfn_checkpoint['net'])

    dpi_checkpoint = torch.load(r'...')
    T_dpi.load_state_dict(dpi_checkpoint['T_dpi'])

    with torch.no_grad():
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": "object detection task", },
            ],
            # [
            #     {"role": "user", "content": "semantic segmentation task", },
            # ],
            # [
            #     {"role": "user", "content": "salient object detection task", },
            # ],
        ]
        textf = llama_generator.chat_completion_forTQFusion(
            dialogs,
            max_gen_len=None,
        )
        textf = torch.mean(textf, dim=1)
    del llama_generator
    torch.cuda.empty_cache()

    save_inf = {
        'save_compare_dir': save_compare_dir,
        "save_eq_fusion_dir": save_eq_fusion_dir,
    }

    test_dataset = TestDataset(vis_test_dir, ir_test_dir, tf_test,
                               img_type='RGB', is_d2=False)

    test_data_iter = data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        generator=torch.Generator(device=device)
    )
    test(bfn, T_dpi, textf, test_data_iter, save_inf)


def test(bfn, T_dpi, textf_gen, test_data_iter, save_inf):
    for step, (vis, ir, file_dir) in tqdm(enumerate(test_data_iter)):
        vi = vis.to(device).bfloat16().detach()  # vis
        ir = ir.to(device).bfloat16().detach()  # ir

        vi_ycbcr = utils.rgb2ycbcr(vi)
        vi_Y, vi_Cb, vi_Cr = torch.split(vi_ycbcr, 1, 1)

        ir = ir[:, :1, :, :]

        with torch.no_grad():
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

        plex = '.png'
        file_name = file_dir[0].split('/')[-1].split('.')[0]

        output_name = save_inf['save_compare_dir'] + "/" + file_name + plex
        out = torch.cat([vi_Y, ir, f_img], dim=2)
        save_image(out, output_name)

        output_name = save_inf['save_eq_fusion_dir'] + "/" + file_name + plex
        out = f_img_c
        save_image(out, output_name)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()