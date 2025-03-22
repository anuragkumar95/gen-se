import os
import json
import torch
import wandb
import argparse
import torchaudio
import torch.nn as nn
from components.encodec.utils import AttrDict
from components.encodec.model import SimCodec, MultiScaleSTFTDiscriminator
from compute_metrics import compute_metrics
from dataset.dataset import LibriSpeechDataset
from torch.utils.data import DataLoader

class SimCodecTrainer:
    def __init__(self, config):
        self.n_fft = 400
        self.hop_length = 100
        self.win_length = self.n_fft
        self.lambda1 = 45
        self.lambda2 = 0.1

        self.training_params = AttrDict(config['training_params'])
        self.device = torch.device(self.training_params.gpu if self.training_params.gpu is not None else 'cpu')

        self.model = SimCodec( config['simcodec_params'] ).to(self.device)
        self.discriminator = MultiScaleSTFTDiscriminator( **config['discriminator_params'] ).to(self.device)

        self.g_optim = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad, self.model.parameters()), lr=self.training_params.learning_rate
        )
        self.d_optim = torch.optim.AdamW(
            filter(lambda layer:layer.requires_grad, self.discriminator.parameters()), lr=2*self.training_params.learning_rate
        )

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=True, center=False, pad_mode=None, power=None
        ).to(self.device)

        wandb.login()
        wandb.init(project=self.training_params.experiment, name=self.training_params.suffix)


    def save(self, epoch, pesq):
        save_dict = {
            'simcodec':self.model.state_dict(),
            'discriminator':self.discriminator.state_dict(),
            'g_optim':self.g_optim.state_dict(),
            'd_optim':self.d_optim.state_dict(),
            'epoch':epoch,
            'pesq':pesq
        }
        exp = self.training_params.experiment
        suf = self.training_params.suffix
        save_dir = self.training_params.save_dir
        _dir_ = f"{save_dir}/{exp}_{suff}"
        os.makedirs(_dir_, exist_ok=True)
        f_id = f"{epoch}_{'0.:4f'.format(pesq)}.pt"
        path = os.path.join(_dir_, f_id)
        torch.save(save_dict, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['simcodec'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.g_optim.load_state_dict(ckpt['g_optim'])
        self.d_optim.load_state_dict(ckpt['d_optim'])
        print(f"Checkpoint loaded from {path} successfully. Found PESQ of {ckpt['pesq']}")

    def codec_forward_one_step(self, batch):
        
        batch_of_wavs, _ = batch
        batch_of_wavs = batch_of_wavs.to(self.device)
        
        #Forward pass through simcodec
        recons, z, q_z, codes, q_loss = self.model(batch_of_wavs)

        return {
            'recons':recons,
            'codes':codes,
            'input':batch_of_wavs,
            'z':z,
            'q_z':q_z,
            'q_loss':q_loss
        }

    def calculate_gen_loss(self, gen_outputs):
        x = gen_outputs['input']
        x_pred = gen_outputs['recons']

        #Transform
        spec_x = torch.view_as_real(self.spec_transform(x))
        spec_x_pred = torch.view_as_real(self.spec_transform(x_pred))

        #Reconstruction loss
        re_loss = (spec_x - spec_x_pred).abs().mean() + ((spec_x - spec_x_pred)**2).mean() + (x - x_pred).abs().mean()

        #Commit loss
        #commit_loss = ((gen_outputs['z'] - gen_outputs[q_z])**2).mean()
        commit_loss = gen_outputs['q_loss']
        #quant_loss = gen_outputs['q_loss']

        #generator loss
        d_xpred_logits, d_xpred_fmaps = self.discriminator(x_pred)
        d_xpred = torch.mean(d_xpred_logits, dim=-1)
        gen_loss = ((d_xpred - gen_outputs['one_labels'])**2).mean()

        #feature_loss
        _, d_x_fmaps = self.discriminator(x)
        feature_loss = 0
        for fm_x, fm_xpred in zip(d_x_fmaps, d_xpred_fmaps):
            for f_x, f_xpred in zip(fm_x, fm_xpred):
                feature_loss += (f_x - f_xpred).abs().mean()

        return {
            're_loss':re_loss,
            'commit_loss':commit_loss,
            'gen_loss':gen_loss,
            'feature_loss':feature_loss,
        }

    def disc_forward_one_step(self, gen_outputs):
        x = gen_outputs['input']
        x_pred = gen_outputs['recons'].detach()

        #discriminator forwardpass
        d_x_logits, d_x_fmaps = self.discriminator(x)
        d_xpred_logits, d_xpred_fmaps = self.discriminator(x_pred)

        d_x = torch.mean(d_x_logits, dim=-1)
        d_xpred = torch.mean(d_xpred_logits, dim=-1) 

        return {
            'd_x':d_x,
            'd_xpred':d_xpred,
        }       

    def calculate_disc_loss(self, disc_outputs):
        d_x = disc_outputs['d_x']
        d_xpred = disc_outputs['d_xpred']

        d_loss = ((d_x - disc_outputs['one_labels'])**2).mean() + d_xpred.mean()
        return d_loss

    def run_metrics(self, clean, recons):
        clean = clean.detach().cpu().numpy()
        recons = recons.detach().cpu().numpy()

        metrics = {
            'pesq':0,
            'csig':0,
            'cbak':0,
            'covl':0,
            'segSNR':0,
            'stoi':0,
            'sisdr':0
        }

        for i in range(clean.shape[0]):
            pesq, csig, cbak, covl, segSNR, stoi, sisdr = compute_metrics(
                clean[i, ...].reshape(-1), recons[i, ...].reshape(-1), 16000, 0
            )
            metrics['pesq'] += pesq/clean.shape[0]
            metrics['csig'] += csig/clean.shape[0]
            metrics['cbak'] += cbak/clean.shape[0]
            metrics['covl'] += covl/clean.shape[0]
            metrics['segSNR'] += segSNR/clean.shape[0]
            metrics['stoi'] += stoi/clean.shape[0]
            metrics['sisdr'] += sisdr/clean.shape[0]

        return metrics


    def train_one_epoch(self, train_dataloader):

        self.model.train()
        self.discriminator.train()

        train_ds_len = len(train_dataloader)
        
        for step, batch in enumerate(train_dataloader):
            
            ############################GENERATOR STEP################################

            gen_outputs = self.codec_forward_one_step(batch)
            gen_outputs['one_labels'] = torch.ones(self.training_params.batchsize).to(self.device)
            gen_loss = self.calculate_gen_loss(gen_outputs)
            G_LOSS = self.lambda1 * gen_loss['re_loss'] + \
                     self.lambda2 * gen_loss['commit_loss'] + \
                     gen_loss['gen_loss'] + gen_loss['feature_loss']

            if torch.isnan(G_LOSS) or torch.isinf(G_LOSS):
                continue
            
            G_LOSS = G_LOSS / self.training_params.accum_grad
            G_LOSS.backward()

            if (step+1) % self.training_params.accum_grad == 0 or (step+1) == train_ds_len:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.g_optim.step()
                self.g_optim.zero_grad()

            ############################DISCRIMINATOR STEP################################    
            
            disc_outputs = self.disc_forward_one_step(gen_outputs)
            disc_outputs['one_labels'] = torch.ones(self.training_params.batchsize).to(self.device)
            D_LOSS = self.calculate_disc_loss(disc_outputs)

            if torch.isnan(D_LOSS) or torch.isinf(D_LOSS):
                continue

            D_LOSS = D_LOSS / self.training_params.accum_grad
            D_LOSS.backward()   

            if (step+1) % self.training_params.accum_grad == 0 or (step+1) == train_ds_len:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
                self.d_optim.step()
                self.d_optim.zero_grad()

            
            print(f"G_LOSS:{G_LOSS.item()} D_LOSS:{D_LOSS.item()}")

            if (step+1) % self.training_params.accum_grad == 0 or (step+1) == train_ds_len:
                wandb.log({
                    'train_G_LOSS':G_LOSS.item(),
                    'train_D_LOSS':D_LOSS.item(),
                    'train_gen_loss':gen_loss['gen_loss'].item(),
                    'train_feature_loss':gen_loss['feature_loss'].item(),
                    'train_re_loss':gen_loss['re_loss'].item(),
                    'train_commit_loss':gen_loss['commit_loss'].item(),
                })


    def validation_one_epoch(self, valid_dataloader):

        self.model.eval()
        self.discriminator.eval()
        
        metrics = {
            'pesq':0,
            'csig':0,
            'cbak':0,
            'covl':0,
            'segSNR':0,
            'stoi':0,
            'sisdr':0
        }

        valid_ds_len = len(valid_dataloader)

        for step, batch in enumerate(valid_dataloader):
            gen_outputs = self.codec_forward_one_step(batch)
            gen_outputs['one_labels'] = torch.ones(self.training_params.batchsize).to(self.device)
            gen_loss = self.calculate_gen_loss(gen_outputs)
            G_LOSS = self.lambda1 * gen_loss['re_loss'] + \
                     self.lambda2 * gen_loss['commit_loss'] + \
                     gen_loss['gen_loss'] + gen_loss['feature_loss']

            disc_outputs = self.disc_forward_one_step(gen_outputs)
            disc_outputs['one_labels'] = torch.ones(self.training_params.batchsize).to(self.device)
            D_LOSS = self.calculate_disc_loss(disc_outputs)

            batch_metrics = self.run_metrics(gen_outputs['input'], gen_outputs['recons'])

            for key in batch_metrics.keys():
                metrics[key] += batch_metrics[key] / valid_ds_len

        wandb.log({
            'val_G_LOSS':G_LOSS.item(),
            'val_D_LOSS':D_LOSS.item(),
            'val_gen_loss':gen_loss['gen_loss'].item(),
            'val_feature_loss':gen_loss['feature_loss'].item(),
            'val_re_loss':gen_loss['re_loss'].item(),
            'val_commit_loss':gen_loss['commit_loss'].item(),
            'val_pesq':metrics['pesq'],
            'val_csig':metrics['csig'],
            'val_cbak':metrics['cbak'],
            'val_covl':metrics['covl'],
            'val_segSNR':metrics['segSNR'],
            'val_stoi':metrics['stoi'],
            'val_sisdr':metrics['sisdr']
        })

        return metrics

    def train(self, train_dataloader, valid_dataloader):
        best_pesq = 0
        for ep in range(self.training_params.epochs):
            #Train one epoch
            self.train_one_epoch(train_dataloader)

            #Validate one epoch
            metrics = self.validation_one_epoch(valid_dataloader)
            
            #Save checkpoint
            if best_pesq <= metrics['pesq']:
                best_pesq = metrics['pesq']
                self.save(ep, best_pesq)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Directory to config file.")
    return parser

def main(args):
    
    #Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    #Load dataset
    train_dataset = LibriSpeechDataset(
        root = config['data_params']['train_root'],
        cutlen_dur = config['data_params']['cutlen_dur_sec'],
    )
    valid_dataset = LibriSpeechDataset(
        root = config['data_params']['val_root'],
        cutlen_dur = config['data_params']['cutlen_dur_sec'],
    )
    
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config['training_params']['batchsize'], 
        shuffle=True,
        drop_last=False
    )
    valid_dl = DataLoader(
        valid_dataset, 
        batch_size=config['training_params']['batchsize'], 
        shuffle=True,
        drop_last=False
    )

    print(f"Datasets loaded successfully. Train dataset length: {len(train_dl)} Valid dataset length: {len(valid_dl)}")

    #Initialize trainer
    trainer = SimCodecTrainer(config)
    trainer.train(train_dl, valid_dl)



if __name__=='__main__':
    ARGS = args().parse_args()
    main(ARGS)

