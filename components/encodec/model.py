from components.encodec.modules import (
    SEANetEncoder, 
    SEANetDecoder, 
    DiscriminatorSTFT, 
    DiscriminatorOutput, 
    Quantizer
)
import torch.nn as nn
import typing as tp
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SimCodec(nn.Module):
    def __init__(self, config, add_linear=False):
        super(SimCodec, self).__init__()
        
        self.encoder = SEANetEncoder(
            dimension=config['dim']
        )

        self.quantizer = Quantizer(
            vq_dim=config['dim'], 
            n_codes=config['n_codes'], 
            n_code_groups=config['n_groups'], 
            n_residual=1,
            codebook_loss_lambda=config['codebook_loss_lambda'],
            commitment_loss_lambda=config['commitment_loss_lambda']
        )
        
        self.generator = SEANetDecoder(
            dimension=config['dim']
        )
        
        self.linear_projection = add_linear
        if add_linear:
            self.enc_proj = nn.Linear(config['dim'], 2 * config['dim'])
            self.dec_proj = nn.Linear(2 * config['dim'], config['dim'])
    
    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path,map_location='cpu')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        self.generator.load_state_dict(ckpt['generator'])

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x) 
        if self.linear_projection:
            c = self.enc_proj(c.permute(0, 2, 1)).permute(0, 2, 1)
        z_c, q_loss, c_i = self.quantizer(c)
        c_i = [code.reshape(batch_size, -1) for code in c_i]

        if self.linear_projection:
            z_c = self.dec_proj(z_c.permute(0, 2, 1)).permute(0, 2, 1)

        recons = self.generator(z_c)
        return recons, z_c, c, torch.stack(c_i, -1), q_loss

    def decode(self, x):
        x = self.quantizer.embed(x)
        if self.linear_projection:
            x = self.dec_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.generator(x)


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512], hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logit = torch.mean(logit, dim=[1,2,3]).reshape(-1, 1)
            logits.append(logit.mean())
            fmaps.append(fmap)

        logits = torch.stack(logits, dim=-1)
        
        return logits, fmaps
