import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from AutoEncoder.model.DSCBCT_s3 import DSCBCT_s3
from train.callbacks import VolumeLogger
from omegaconf import DictConfig, open_dict
from pytorch_lightning.loggers import TensorBoardLogger
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
from dataset.Stage2_3 import DenoisingDataset
from omegaconf import OmegaConf
import argparse
import torchio as tio 
import torch 
from omegaconf import OmegaConf
import numpy as np


def resize_to_even(image: torch.Tensor) -> torch.Tensor:

    shape = np.array(image.shape[-3:])  
    new_shape = (shape + 1) // 2 * 2  
    return torch.nn.functional.interpolate(image, size=tuple(new_shape), mode='trilinear', align_corners=False), shape

def eval(args):
    output_dir = args.output_dir 
    ckpt_path = args.ckpt_path 
    input_dir = args.input_dir 
    os.makedirs(output_dir,exist_ok=True)
    dataset = DenoisingDataset(input_dir= input_dir, split='test')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model = DSCBCT_s3.load_for_inference(checkpoint_path=ckpt_path, is_inference=True).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in test_dataloader:
        sample, affine, path = batch['input'].to(device), batch['affine'], batch['path']
        resized_sample, original_shape = resize_to_even(sample)
        with torch.no_grad():
            output = model.inference_sliding_new(resized_sample)

        output_ = (output[0] + 1) / 2
        output_ = torch.clamp(output_, min=0, max=1)
        output_ = torch.nn.functional.interpolate(output_.unsqueeze(0), size=tuple(original_shape), mode='trilinear', align_corners=False).squeeze(0)
        output_ = output_.cpu().transpose(2, 3).transpose(1, 3)
        out_img = tio.ScalarImage(tensor=output_, affine=affine[0].squeeze(0))
        out_img.save(os.path.join(output_dir, os.path.basename(path[0])))

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)

    args = parser.parse_args()
    eval(args)