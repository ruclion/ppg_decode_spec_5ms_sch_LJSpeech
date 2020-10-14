# 不会的问题：GriffinLim的power和iter不会设置，也不知道有什么影响，长河的1.0和100，常用的是1.5和60
import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model_torch import DCBHG
from dataload_ljspeech import ljspeechDtaset
from audio import hparams as audio_hparams
from audio import normalized_db_mel2wav, normalized_db_spec2wav, write_wav


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


# 用GPU训练，CPU读数据
use_cuda = torch.cuda.is_available()
assert use_cuda is True
device = torch.device("cuda" if use_cuda else "cpu")
num_workers = 1

# some super parameters，用epochs来计数，而不是步数（不过两者同时统计）
BATCH_SIZE = 64
clip_thresh = 0.1
# BATCH_SIZE = 16
nepochs=5000,
LEARNING_RATE = 0.0003
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
CKPT_EVERY = 150

# ljspeech的log和ckpt文件夹
ljspeech_log_dir = os.path.join('ljspeech_log_dir', STARTED_DATESTRING, 'train_and_dev')
ljspeech_model_dir = os.path.join('ljspeech_log_dir', STARTED_DATESTRING, 'ckpt_model')
if os.path.exists(ljspeech_log_dir) is False:
  os.makedirs(ljspeech_log_dir, exist_ok=True)
if os.path.exists(ljspeech_model_dir) is False:
  os.makedirs(ljspeech_model_dir, exist_ok=True)


# 计数全局变量
global_step = 0
global_epoch = 0



# 恢复训练，还没写完，不能用 TODO
# def load_model(model, ckpt_path):
#   global_step
#   global_epch
#   ckpt_checkpoint_load = torch.load(ckpt_path)
#   model.load_state_dict(ckpt_checkpoint_load["state_dict"])
#   return model


def eval_model_generate(spec, spec_pred, length, log_dir, global_step):
  print("EVAL LENGTH:", length)
  print("EVAL SPEC PRED SHAPE:", spec_pred.shape)
  
  y_pred = normalized_db_spec2wav(spec_pred)
  pred_wav_path = os.path.join(log_dir, "checkpoint_step_{}_pred.wav".format(global_step))
  write_wav(pred_wav_path, y_pred)
  pred_spec_path = os.path.join(log_dir, "checkpoint_step_{}_pred_spec.npy".format(global_step))
  np.save(pred_spec_path, spec_pred)


  print("EVAL LENGTH:", length)
  print("EVAL SPEC SHAPE:", spec.shape)
  y = normalized_db_spec2wav(spec)
  orig_wav_path = os.path.join(log_dir, "checkpoint_step_{}_original.wav".format(global_step))
  write_wav(orig_wav_path, y)
  orig_spec_path = os.path.join(log_dir, "checkpoint_step_{}_orig_spec.npy".format(global_step))
  np.save(orig_spec_path, spec)


def main():
  # 数据读入，准备
  now_dataset = ljspeechDtaset()
  now_torch_dataloader = DataLoader(now_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last=True)


  # 构建模型，放在gpu上，顺便把tensorboard的图的记录变量操作也算在这里面
  model = DCBHG().to(device)
  writer = SummaryWriter(log_dir=ljspeech_log_dir)


  # 设置梯度回传优化器，目前使用固定lr=0.0003，不知用不用变lr
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  

  # optimize classification
  # cross_entropy_loss = nn.CrossEntropyLoss()
  # criterion = nn.MSELoss()
  # l1_loss = nn.NLLLoss()
  # from kuaishou 
  my_l1_loss = nn.L1Loss()


  # 开始训练
  print('开始训练')
  global global_step, global_epoch
  model.train()

  while global_epoch < nepochs:
      running_loss = 0.0
      for _step, (ppgs, mels, specs, lengths) in tqdm(enumerate(now_torch_dataloader)):
          # Batch开始训练，清空opt，数据拿到GPU上
          optimizer.zero_grad()

          ppgs = ppgs.to(device)
          mels = mels.to(device)
          specs = specs.to(device)
          print('before type:', type(ppgs), type(mels), type(specs))
          ppgs, mels, specs = Variable(ppgs).float(), Variable(mels).float(), Variable(specs).float()
          print('after type:', type(ppgs), type(mels), type(specs))


          # Batch同时计算出pred结果
          mels_pred, specs_pred = model(ppgs)


          # 根据预测结果定义/计算loss; 不过我记得tacotron里面不是用的两个l1loss吧，之后再看看 TODO
          loss = 0.0
          for i in range(BATCH_SIZE):
            mel_loss = my_l1_loss(mels_pred[i, :lengths[i], :], mels[i, :lengths[i], :])
            spec_loss = my_l1_loss(specs_pred[i, :lengths[i], :], specs[i, :lengths[i], :])
            loss += (mel_loss + spec_loss)
          loss = loss / BATCH_SIZE
          print("Check Loss：", loss)
          writer.add_scalar("loss", float(loss.item()), global_step)
          running_loss += loss.item() # 计算epoch的平均loss累计值


          # 根据loss，计算梯度，并且应用梯度回传操作，改变权重值
          loss.backward()
          if clip_thresh > 0:
            _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh) # 返回值不用管
          optimizer.step()


          # 存储ckpt，并且同样的步数，存储生成的音频
          if global_step > 0 and global_step % CKPT_EVERY == 0:
            checkpoint_path = os.path.join(ljspeech_model_dir, "checkpoint_step{:09d}.pth".format(global_step))
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "global_epoch": global_epoch,
            }, checkpoint_path)
            eval_model_generate(specs[0].cpu().data.numpy(), specs_pred[0].cpu().data.numpy(), lengths[0], ljspeech_log_dir, global_step)
          

          # BATCH操作结束，step++
          global_step += 1

      # 开始对整个epoch进行信息统计
      averaged_loss = running_loss / (len(now_torch_dataloader))
      writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
      global_epoch += 1


if __name__ == '__main__':
  main()
