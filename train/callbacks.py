

import os
import numpy as np
from PIL import Image

import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torchio as tio





class VolumeLogger(Callback):
    def __init__(self, batch_frequency, max_volumes, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_volumes = max_volumes
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, volumes,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "volumes", split)

        
        for k in volumes:
            volumes[k] = (volumes[k] + 1.0)/2.0
            for idx,volume in enumerate(volumes[k]):
                volume = volume.transpose(1,3).transpose(1,2)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}-{}.nii.gz".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                tio.ScalarImage(tensor = volume).save(path)
         
    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_volumes") and
                callable(pl_module.log_volumes) and
                self.max_volumes > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                volumes = pl_module.log_volumes(
                    batch, split=split, batch_idx=batch_idx)

            for k in volumes:
                N = min(volumes[k].shape[0], self.max_volumes)
                volumes[k] = volumes[k][:N]
                if isinstance(volumes[k], torch.Tensor):
                    volumes[k] = volumes[k].detach().cpu()

            self.log_local(pl_module.logger.save_dir, split, volumes,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx ):
        self.log_vid(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_vid(pl_module, batch, batch_idx, split="val")
