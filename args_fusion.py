
import argparse

parser = argparse.ArgumentParser(description='****')
parser.add_argument('--train_save_img_dir', default='./checkpoints/images', type=str)
parser.add_argument('--train_save_model_dir', default='./checkpoints/train_models', type=str)
parser.add_argument('--path_checkpoint', default='...', type=str)
parser.add_argument('--save_image_num', dest='save_image_num', default=8, type=int)
parser.add_argument('--save_model_num', dest='save_model_num', default=20, type=int)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--batch_size', dest='batch_size', default=6, type=int)
parser.add_argument('--LR', type=float, default=0.0001)
parser.add_argument('--LR_target', type=float, default=0.001)
parser.add_argument('--Epoch', type=float, default=100)
parser.add_argument('--Warm_epoch', type=float, default=20)
args = parser.parse_args()