from torch.utils.data import Dataset
import torchaudio
import torch
import torchaudio.functional as F
import torch.nn.functional as Fn
import random
from torch.distributions.uniform import Uniform
import glob
from pathlib import Path 
from tqdm import tqdm

def mix_audios(clean, noise, snr):
        
    assert clean.shape[-1] == noise.shape[-1]

    #calculate the amount of noise to add to get a specific snr
    p_clean = (clean ** 2).mean().reshape(-1)
    p_noise = (noise ** 2).mean().reshape(-1)

    p_ratio = p_clean / p_noise
    alpha = torch.sqrt(p_ratio / (10 ** (snr / 10))).reshape(-1, 1)
    signal = clean + (alpha * noise)
    
    return signal


class SimCodecDataset(Dataset):
    """
    This dataset requires manifest files containing full paths to all 
    audio files. 
    """
    def __init__(self, clean_manifest, noise_manifest, rir_manifest, cutlen_dur_sec=5):
        self.clean_paths = [path.strip() for path in open(clean_manifest).readlines()]
        self.noise_paths = [path.strip() for path in open(noise_manifest).readlines()]
        self.rir_paths = [path.strip() for path in open(rir_manifest).readlines()]
        self.n_noise = len(self.noise_paths)
        self.n_rir = len(self.rir_paths)
        self.cutlen = 16000 * cutlen_dur_sec

        print(f"CLEAN:{len(self.clean_paths)}, NOISE:{len(self.noise_paths)}, RIR:{len(self.rir_paths)}")

    def __len__(self, ):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        r_idx_noise = random.randint(0, self.n_noise-1)
        r_idx_rir = random.randint(0, self.n_rir-1)
        
        clean, c_sr = torchaudio.load(self.clean_paths[idx])
        noise, n_sr = torchaudio.load(self.noise_paths[r_idx_noise])
        rir, r_sr = torchaudio.load(self.rir_paths[r_idx_rir])

        if max(c_sr, n_sr, r_sr) != 16000 or min(c_sr, n_sr, r_sr) != 16000:
            raise ValueError("Sampling rate other than 16k found.")

        n_ch = random.randint(0, 100)
        r_ch = random.randint(0, 100)

        #Pad or crop audio
        if clean.shape[-1] < self.cutlen:
            clean = Fn.pad(clean, (0, self.cutlen - clean.shape[-1]))
        else:
            clean = clean[:, :self.cutlen]
        
        noisy = clean
        CASE = -1
        if n_ch >=20:
            CASE=1
            snr = Uniform(-5, 20).sample()
            while noise.shape[-1] <= clean.shape[-1]:
                noise = torch.cat([noise, noise], dim=-1)
            noise = noise[:, :clean.shape[-1]]

            assert noise.shape[-1] == clean.shape[-1], f"NOISE:{noise.shape}, CLEAN:{clean.shape}"
            noisy = mix_audios(clean, noise, snr)

        if r_ch >=50:
            CASE=2
            rir = rir / torch.linalg.vector_norm(rir, ord=2)
            noisy = F.fftconvolve(noisy, rir)
            noisy = noisy[:, :clean.shape[-1]]

        return clean, noisy 


class LibriSpeechDataset(Dataset):
    def __init__(self, root, cutlen_dur=5, extn='flac'):
        self.paths = {
            Path(i).stem:{'audio':i, 'trans':''} for i in glob.glob(f"{root}/**/*.{extn}", recursive=True)
        }

        print(f"Compiling transcripts...")
        trans_path = glob.glob(f"{root}/**/*.txt", recursive=True)
        for t_path in tqdm(trans_path):
            with open(t_path, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    f_id = line[0]
                    transcript = " ".join(line[1:])
                    self.paths[f_id]['trans'] = transcript

        self.keys = list(self.paths.keys())
        self.cutlen = 16000 * cutlen_dur  


    def __len__(self, ):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.paths[key]

        audio, fs = torchaudio.load(path['audio'])
        
        #Pad or crop audio
        if audio.shape[-1] <= self.cutlen:
            audio = Fn.pad(audio, (0, self.cutlen - audio.shape[-1]))
        else:
            try:
                ridx = random.randint(0, audio.shape[-1] - self.cutlen - 1)
                audio = audio[:, ridx : ridx + self.cutlen]
            except:
                print(f"IDX:{idx}, SHAPE:{audio.shape}, CUTLEN:{self.cutlen}")
    
        return audio, path['trans']


 