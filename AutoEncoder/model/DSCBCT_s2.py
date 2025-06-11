

import math
import argparse
import numpy as np
import pickle as pkl
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from AutoEncoder.model.lpips import LPIPS
from AutoEncoder.model.VQGAN import VQGAN
from AutoEncoder.model.CodeClassifier import CodeClassifier
from AutoEncoder.model.codebook import Codebook
from einops_exts import  rearrange_many
def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)



def check_param(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        assert torch.equal(param1.cuda(), param2.cuda())
def freeze_parameter(net):
    for param in net.parameters():
        param.requires_grad = False

class DSCBCT_s2(pl.LightningModule):
    def __init__(self, cfg, is_inference=False):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type,
                               cfg.model.num_groups,cfg.model.embedding_dim,
                               )
        self.pretrained_encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type,
                               cfg.model.num_groups,cfg.model.embedding_dim,
                               )
        
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups,cfg.model.embedding_dim)
        self.enc_out_ch = self.encoder.out_channels

        self.pre_vq_conv = nn.Conv3d(cfg.model.embedding_dim, cfg.model.embedding_dim, 1, 1)
        self.pretrained_pre_vq_conv = nn.Conv3d(cfg.model.embedding_dim, cfg.model.embedding_dim, 1, 1)

        self.post_vq_conv = nn.Conv3d(cfg.model.embedding_dim, cfg.model.embedding_dim, 1, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)
        self.codeCNN = CodeClassifier(latent_size=cfg.model.embedding_dim, dim_embd=cfg.model.dim_embd, num_layers=cfg.model.n_layers, codebook_size=cfg.model.n_codes, num_groups=32)

        if not is_inference and cfg.model.resume_from_checkpoint is None:
            self._load_pretrained_vqgan(cfg)

        freeze_parameter(self.decoder)
        freeze_parameter(self.post_vq_conv)
        freeze_parameter(self.codebook)
        freeze_parameter(self.pretrained_encoder)
        freeze_parameter(self.pretrained_pre_vq_conv)
        
    
        self.feat_loss_weight = cfg.model.feat_loss_weight
        self.ce_loss_weight = cfg.model.ce_loss_weight
        self.l1_weight = cfg.model.l1_weight
        self.save_hyperparameters()

    def _load_pretrained_vqgan(self, cfg):
            self.pretrain_model = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt).eval()
            self.encoder.load_state_dict(self.pretrain_model.encoder.state_dict())
            self.pretrained_encoder.load_state_dict(self.pretrain_model.encoder.state_dict())
            self.pretrained_encoder.eval()
            self.codebook.load_state_dict(self.pretrain_model.codebook.state_dict())
            self.pre_vq_conv.load_state_dict(self.pretrain_model.pre_vq_conv.state_dict())
            self.pretrained_pre_vq_conv.load_state_dict(self.pretrain_model.pre_vq_conv.state_dict())
            self.pretrained_pre_vq_conv.eval()
            self.post_vq_conv.load_state_dict(self.pretrain_model.post_vq_conv.state_dict())
            self.decoder.load_state_dict(self.pretrain_model.decoder.state_dict())
            del self.pretrain_model

    def forward(self, x, gt=None,  log_volume=False,val=False):
        B, C, D, H, W = x.shape ##ｂ　ｃ　ｚ　ｘ　ｙ
        
        z = self.pre_vq_conv(self.encoder(x))
        vq_encodings = self.codeCNN(z)
        if not val and not log_volume :  ## generate gt
            z_gt = self.pretrained_pre_vq_conv(self.pretrained_encoder(gt))
            encodings_gt = self.codebook(z_gt)['encodings']
            feat_encoder_loss = torch.mean((z_gt.detach()-z)**2) * self.feat_loss_weight
            ce_loss = F.cross_entropy(vq_encodings ,encodings_gt.detach() ) * self.ce_loss_weight
            total_loss = ce_loss + feat_encoder_loss
            self.log("train/ce_loss", ce_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/feat_encoder_loss", feat_encoder_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/total_loss", total_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return total_loss
        else:
            soft_one_hot = F.softmax(vq_encodings, dim=1)
            _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
            vq = F.embedding(top_idx.squeeze(1), self.codebook.embeddings)
            vq = vq.permute(0,4,1,2,3)
            x_recon = self.decoder(self.post_vq_conv(vq))
            recon_loss = F.l1_loss(x_recon, gt) * self.l1_weight
            return recon_loss,x_recon
    def encode(self, x,):
        z = self.pre_vq_conv(self.encoder(x))
        return z


    def inference_sliding(self, image_data, overlap=16):
        
        latent = self.pre_vq_conv(self.encoder(image_data))
        B, C, H, W, D = latent.shape


        window_size = 128//4
        stride = window_size-overlap//4



        latent_output = torch.zeros(latent.shape[0],1024,latent.shape[2],latent.shape[3],latent.shape[4]).cuda()
        count_map = torch.zeros_like(latent_output)  # 用来平均重叠区域



        for x in range(0, H, stride):
            for y in range(0, W, stride):
                for z in range(0, D, stride):
                    # 取出滑动窗口的 patch
                    x_start , y_start , z_start = min(H - window_size , x ) , min(W - window_size , y ), min(D - window_size , z)
                    patch = latent[:,:, x_start:x_start+window_size, y_start:y_start+window_size, z_start:z_start+window_size]


                    vq_encodings = self.codeCNN(patch)
                    latent_output[:, :, x_start:x_start+window_size, y_start:y_start+window_size, z_start:z_start+window_size] += vq_encodings
                    count_map[:, :, x_start:x_start+window_size, y_start:y_start+window_size, z_start:z_start+window_size] += 1
        latent_output /= count_map



        soft_one_hot = F.softmax(latent_output, dim=1)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
        
        # 查找 codebook embedding
        vq = F.embedding(top_idx.squeeze(1), self.codebook.embeddings)
        vq = vq.permute(0, 4, 1, 2, 3)
        
        # 解码
        x_recon = self.decoder(self.post_vq_conv(vq))



        return x_recon


    
    def training_step(self, batch, batch_idx):
        x , gt = batch['input'] , batch['target']
        total_loss = self.forward(x,gt)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x  , gt = batch['input'], batch['target']  
        recon_loss,_ = self.forward(x,gt= gt, val=True)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.pre_vq_conv.parameters())+
                                  list(self.codeCNN.parameters()), 
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae, ], []



    def log_volumes(self, batch, **kwargs):
        log = dict()
        x , gt = batch['input'] , batch['target']
        x , gt = x.to(self.device) , gt.to(self.device)
        _, x_recon = self(x, gt=gt,log_volume=True, val=(kwargs['split']=='val'))
        log["inputs"] = x
        log["gt"] = gt
        log["reconstructions"] = x_recon

        return log

    @classmethod
    def load_for_inference(cls, checkpoint_path, is_inference=True):
        """用于加载训练后的模型进行推理"""
        # 通过传递 is_inference 为 True 来避免加载预训练模型
        model = cls.load_from_checkpoint(checkpoint_path, is_inference=is_inference)
        return model


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)




class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32,norm_type='group',num_groups=32):
        super().__init__()
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 256
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm,'b c z x y -> b (z x y) c').contiguous()
        qkv = self.to_qkv(x_norm).chunk(3, dim=2)
        q, k, v = rearrange_many(
            qkv, 'b d (h c) -> b h d c ', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y ',z = z, x = h ,y = w ).contiguous()
        out = self.to_out(out)
        return out+x


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=1, norm_type='group', num_groups=32 , embedding_dim = 8):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(
            image_channel , n_hiddens, kernel_size=3, stride=1, padding=1
        )
    
        channels = [n_hiddens * 2 ** i for i in range(max_ds)]
        channels = channels +[channels[-1]]
        in_channels = channels[0]
        for i in range(max_ds + 1):
            block = nn.Module()
            if i != 0 :
                in_channels = channels[i-1]
            out_channels = channels[i]
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            if in_channels!= out_channels:
                block.res1 = ResBlockXY(in_channels , out_channels,norm_type=norm_type, num_groups=num_groups )
            else:
                block.res1 = ResBlockX(in_channels , out_channels,norm_type=norm_type, num_groups=num_groups)

            block.res2  = ResBlockX(out_channels , out_channels, norm_type=norm_type, num_groups=num_groups)
            if i != max_ds:
                block.down = nn.Conv3d(out_channels,out_channels,kernel_size=(4, 4, 4),stride=stride,padding=1)
            else:
                block.down = nn.Identity()
            self.conv_blocks.append(block)
            n_times_downsample -= 1
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(out_channels , out_channels,norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(out_channels, heads=4,norm_type=norm_type,num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(out_channels , out_channels,norm_type=norm_type, num_groups=num_groups)
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, self.embedding_dim, 3 , 1 ,1)
        )

        self.out_channels = out_channels
    def forward(self, x):
        h = self.conv_first(x)
        for idx , block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.down(h)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        h = self.final_block(h)
        return h

class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32 , embedding_dim=8 ):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        channels = [n_hiddens * 2 ** i for i in range(max_us)]
        channels = channels+[channels[-1]]
        channels.reverse()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(self.embedding_dim, channels[0],3,1,1)
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(channels[0] , channels[0],norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(channels[0], heads=4,norm_type=norm_type,num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(channels[0] , channels[0],norm_type=norm_type, num_groups=num_groups)
        self.conv_blocks = nn.ModuleList()
        in_channels = channels[0]
        for i in range(max_us + 1):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i-1]
            out_channels = channels[i]
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            if in_channels != out_channels:
                block.res1 = ResBlockXY(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            else:
                block.res1 = ResBlockX(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlockX(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            if i != max_us :
                block.up = Upsample(out_channels)
            else:
                block.up = nn.Identity(out_channels)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, image_channel, 3 , 1 ,1)
        )

    def forward(self, x):
        h = self.conv_first(x)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        for i, block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.up(h)
        h = self.final_block(h)
        return h



class ResBlockX(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, norm_type='group', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3 , padding=1, stride=1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        return x+h


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, in_channels, 4,
                                        stride=2, padding=1)

    def forward(self, x):
        x = self.conv_trans(x)
        return x

class ResBlockXY(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, norm_type='group', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.resConv = nn.Conv3d(in_channels, out_channels, (1, 1, 1)) 
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels , out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels , out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        residual = self.resConv(x)
        h = self.norm1(x)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        return h+residual


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer = nn.SyncBatchNorm):
        super().__init__()

        self.leakyRELU = nn.LeakyReLU()
        self.pool = nn.AvgPool3d((2, 2, 2))
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.resConv = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.resConv(self.pool(x))
        x = self.conv1(x)
        x = self.leakyRELU(x)

        x = self.pool(x) 

        x = self.conv2(x)
        x = self.leakyRELU(x)

        return (x+residual)/math.sqrt(2)
