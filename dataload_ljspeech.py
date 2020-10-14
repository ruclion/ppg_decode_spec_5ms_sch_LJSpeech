import os
import numpy as np
from torch.utils.data import Dataset



TRAIN_FILE = './LibriSpeech/meta.txt'
# TEST_FILE = './LibriSpeech/test_diff_meta_960.txt'
MFCC_DIR =  './LibriSpeech/MFCCs'
PPG_DIR =   './LibriSpeech/PPGs'
MEL_DIR =  './LibriSpeech/MELs'
SPEC_DIR =  './LibriSpeech/SPECs'
max_length = 1000
PPG_DIM = 345
MEL_DIM = 80
SPEC_DIM = 201



def text2list_ljspeech(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.split('|')[0])
    return file_list


def get_single_data_pair(fname, ppg_dir, mel_dir, spec_dir):
    assert os.path.isdir(ppg_dir) and os.path.isdir(mel_dir) and os.path.isdir(spec_dir)

    ppg_f = os.path.join(ppg_dir, fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')
    mel_f = os.path.join(mel_dir, fname+'.npy')
    spec_f = os.path.join(spec_dir, fname+'.npy')

    ppg = np.load(ppg_f)
    mel = np.load(mel_f)
    spec = np.load(spec_f)
    assert mel.shape[0] == ppg.shape[0] and mel.shape[0] == spec.shape[0], fname + ' 维度不相等'
    assert mel.shape[1] == MEL_DIM and ppg.shape[1] == PPG_DIM and spec.shape[1] == SPEC_DIM, fname + ' 特征维度不正确'
    return ppg, mel, spec


class ljspeechDtaset(Dataset):
  def __init__(self):
    self.file_list = text2list_ljspeech(file=TRAIN_FILE)
    # 先沿用长河的，所有batch的序列均padding为10000
    self.max_length = max_length

  # 不知道用处，可能是语法，先留着
  # def __len__(self):
  #   assert (len(self.ppgs) == len(self.mels))
  #   assert (len(self.ppgs) == len(self.specs))
  #   return len(self.ppgs)
  
  def __getitem__(self, idx):
    fname = self.file_list[idx]
    ppg, mel, spec = get_single_data_pair(fname, ppg_dir, mel_dir, spec_dir)
    ppg_len = ppg.shape[0]
    print('CHECK ppg_len:', ppg_len)

    # 为什么长河会有这句代码，先放着以供讨论，注释掉
    # if ppg_len > mel_len:
    #     ppg = ppg[:mel_len, :]


    # 此部分先没改
    pad_length = self.max_length - ppg.shape[0]
    if pad_length > 0:
      ppg_padded = np.vstack((ppg, np.zeros((pad_length, PPG_DIM))))
      mel_padded = np.vstack((mel, np.zeros((pad_length, MEL_DIM))))
      spec_padded = np.vstack((spec, np.zeros((pad_length, SPEC_DIM))))
    else:
      print("BIGGER")
      ppg_padded = ppg[:self.max_length, :]
      mel_padded = mel[:self.max_length, :]
      spec_padded = spec[:self.max_length, :]
      ppg_len = self.max_length

    ppg_padded = ppg_padded.astype(np.float64)
    mel_padded = mel_padded.astype(np.float64)
    spec_padded = spec_padded.astype(np.float64)

    return ppg_padded, mel_padded, spec_padded, ppg_len