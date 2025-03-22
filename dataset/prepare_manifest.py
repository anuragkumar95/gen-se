import glob
from tqdm import tqdm
import os
import numpy as np


class Manifest:
    def __init__(
        self, 
        ls_path=None, 
        ll_path=None, 
        vctk_path=None, 
        dns_path=None, 
        wham_path=None, 
        demand_path=None,
        oslr26=None,
        oslr28=None
    ):
        self.wavs = []
        self.noise = []
        self.rir = []

        if ls_path is not None:
            wavs = glob.glob(f"{ls_path}/**/*.flac", recursive=True)
            self.wavs.extend(wavs)
        
        if ll_path is not None:
            wavs = glob.glob(f"{ll_path}/**/*.wav", recursive=True)
            self.wavs.extend(wavs)

        if vctk_path is not None:
            wavs = glob.glob(f"{vctk_path}/**/*.wav", recursive=True)
            self.wavs.extend(wavs)

        if dns_path is not None:
            wavs = glob.glob(f"{dns_path}/**/*.wav", recursive=True)
            self.wavs.extend(wavs)

        if wham_path is not None:
            wavs = glob.glob(f"{wham_path}/**/*.wav", recursive=True)
            self.noise.extend(wavs)

        if demand_path is not None: 
            wavs = glob.glob(f"{demand_path}/**/*.wav", recursive=True)
            self.noise.extend(wavs)

        if oslr26 is not None:
            wavs = glob.glob(f"{oslr26}/**/*.wav", recursive=True)
            self.rir.extend(wavs)

        if oslr28 is not None:
            wavs = glob.glob(f"{oslr28}/**/*.wav", recursive=True)
            self.rir.extend(wavs)

        #Clean audio files
        ind = np.random.choice(len(self.wavs), int(0.8*len(self.wavs)), replace=False)
        v_ind = [i for i in range(len(self.wavs)) if i not in ind]

        self.train_wavs = [self.wavs[i] for i in ind]
        self.valid_wavs = [self.wavs[i] for i in v_ind[:int(0.5*len(v_ind))]]
        self.test_wavs = [self.wavs[i] for i in v_ind[int(0.5*len(v_ind)):]]

        #Noise audio files
        ind = np.random.choice(len(self.noise), int(0.8*len(self.noise)), replace=False)
        v_ind = [i for i in range(len(self.noise)) if i not in ind]

        self.train_noise = [self.noise[i] for i in ind]
        self.valid_noise = [self.noise[i] for i in v_ind[:int(0.5*len(v_ind))]]
        self.test_noise = [self.noise[i] for i in v_ind[int(0.5*len(v_ind)):]]

        #RIR audio files
        ind = np.random.choice(len(self.rir), int(0.8*len(self.rir)), replace=False)
        v_ind = [i for i in range(len(self.rir)) if i not in ind]

        self.train_rir = [self.rir[i] for i in ind]
        self.valid_rir = [self.rir[i] for i in v_ind[:int(0.5*len(v_ind))]]
        self.test_rir = [self.rir[i] for i in v_ind[int(0.5*len(v_ind)):]]

        print(f"Parsed clean audios : {len(self.wavs)}")
        print(f"Parsed noise audios : {len(self.noise)}")
        print(f"Parsed RIR audios : {len(self.rir)}")



    def write(self, path):

        train_path = f"{path}/train"
        valid_path = f"{path}/valid"
        test_path = f"{path}/test"

        os.makedirs(path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        print(f"Writing train manifests to {train_path}") 
        with open(f"{train_path}/clean.manifest", "w") as f:
            for wav in tqdm(self.train_wavs):
                f.write(f"{wav}\n")

        with open(f"{train_path}/noise.manifest", "w") as f:
            for wav in tqdm(self.train_noise):
                f.write(f"{wav}\n")

        with open(f"{train_path}/rir.manifest", "w") as f:
            for wav in tqdm(self.train_rir):
                f.write(f"{wav}\n")

        print(f"Writing valid manifest to {valid_path}")
        with open(f"{valid_path}/clean.manifest", "w") as f:
            for wav in tqdm(self.valid_wavs):
                f.write(f"{wav}\n")

        with open(f"{valid_path}/noise.manifest", "w") as f:
            for wav in tqdm(self.valid_noise):
                f.write(f"{wav}\n")

        with open(f"{valid_path}/rir.manifest", "w") as f:
            for wav in tqdm(self.valid_rir):
                f.write(f"{wav}\n") 

        print(f"Writing test manifest to {test_path}")
        with open(f"{test_path}/clean.manifest", "w") as f:
            for wav in tqdm(self.test_wavs):
                f.write(f"{wav}\n")

        with open(f"{test_path}/noise.manifest", "w") as f:
            for wav in tqdm(self.test_noise):
                f.write(f"{wav}\n")

        with open(f"{test_path}/rir.manifest", "w") as f:
            for wav in tqdm(self.test_rir):
                f.write(f"{wav}\n")

def main():

    #Add respective paths here
    args = {
        'ls_path': "/fs/ess/PAS2301/Data/Speech/LibriSpeech/train-clean-100",
        'demand_path': "/fs/scratch/PAS2301/kumar1109/demand_noise",
        'oslr26': "/fs/scratch/PAS2301/kumar1109/RIR/openSLR26",
    }
    
    save_dir = "/users/PAS2301/kumar1109/SimCodec_DATA_manifests/debug"
    os.makedirs(save_dir, exist_ok=True)

    writer = Manifest(**args)
    writer.write(save_dir)

if __name__ == "__main__":
    main()