import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import torch
from torch.autograd import Variable

from audio import hparams as audio_hparams
from audio import normalized_db_mel2wav, normalized_db_spec2wav, write_wav

from model_torch import DCBHG


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': False,
}

assert hparams == audio_hparams

# 用GPU预测
use_cuda = torch.cuda.is_available()
assert use_cuda is True

# 超参数和路径
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
ckpt_path_ljspeech = './checkpoints-dblstm/with_l1_loss_500/checkpoint_step000010050.pth'

ppgs_paths = 'syn_ppgs.txt'
ljspeech_log_dir = os.path.join('predict_ljspeech_log_dir', STARTED_DATESTRING)
if os.path.exists(ljspeech_log_dir) is False:
    os.makedirs(ljspeech_log_dir, exist_ok=True)

# 全局变量
model = DCBHG()

def tts_load(model, ckpt_path):
    ckpt_load = torch.load(ckpt_path)
    model.load_state_dict(ckpt_load["state_dict"])
    if use_cuda:
        model = model.cuda()
    model.eval()
    return model


def tts_predict(model, ppg):
    # 准备输入的数据并转换到GPU
    ppg = Variable(torch.from_numpy(ppg)).unsqueeze(0).float()
    if use_cuda:
        ppg = ppg.cuda()

    # 进行预测并数据转换到CPU
    mel_pred, spec_pred = model(ppg)
    mel_pred = mel_pred[0].cpu().data.numpy()
    spec_pred = spec_pred[0].cpu().data.numpy()

    # vocoder合成音频波形文件
    mel_pred_audio = normalized_db_mel2wav(mel_pred)
    spec_pred_audio = normalized_db_spec2wav(spec_pred)

    return mel_pred, spec_pred, mel_pred_audio, spec_pred_audio


def draw_spec(a_path, a):
    plt.imshow(a.T[:1000, :], cmap='hot', interpolation='nearest')
    plt.xlabel('frame nums')
    plt.ylabel('spec')
    plt.tight_layout()
    plt.savefig(a_path, format='png')


def main():
    global model
    model = tts_load(model=model, ckpt_path=ckpt_path_ljspeech)


    ppgs_list = open(ppgs_paths, 'r')
    ppgs_list = [i.strip() for i in ppgs_list]
    for idx, ppg_path in tqdm(enumerate(ppgs_list)):
        ppg = np.load(ppg_path)
        mel_pred, spec_pred, mel_pred_audio, spec_pred_audio = tts_predict(model, ppg)

        write_wav(os.path.join(ljspeech_log_dir, "{}_sample_mel.wav".format(idx)), mel_pred_audio)
        write_wav(os.path.join(ljspeech_log_dir, "{}_sample_spec.wav".format(idx)), spec_pred_audio)

        np.save(os.path.join(ljspeech_log_dir, "{}_sample_mel.npy".format(idx)), mel_pred)
        np.save(os.path.join(ljspeech_log_dir, "{}_sample_spec.npy".format(idx)), spec_pred)

        draw_spec(os.path.join(ljspeech_log_dir, "{}_sample_mel.png".format(idx)), mel_pred)
        draw_spec(os.path.join(ljspeech_log_dir, "{}_sample_spec.png".format(idx)), spec_pred)
      
    

if __name__ == "__main__":
    main()