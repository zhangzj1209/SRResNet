import os
import random
import torch
import obspy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary
from SNR import gen_gauss_noise, check_SNR
from config_param import Config
from Max_Min_Normalize import max_min_normalization
from SRResNet import SRResNet
from UNet import UNET
from train import train
from dataset import My_Dataset

seed = 40
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# In[] Basic parameters
main_path = Config().main_path
valid_vs_total = Config().valid_vs_total
batch_size = Config().batch_size
device = Config().device
lr = Config().init_lr
lr_decay = Config().lr_decay
n_epochs = Config().n_epochs
patience = Config().patience

data_dir = Config().data_dir
save_dir = Config().save_dir
fig_dir = Config().fig_dir
predict_data_dir = Config().predict_data_dir
predict_label_dir = Config().predict_label_dir

# In[] Define dataset
data_files = os.listdir(data_dir)
train_files, valid_files = train_test_split(data_files, random_state=10, test_size=valid_vs_total)

print("The number of train datasets is: ", len(train_files))
print("The number of valid datasets is: ", len(valid_files))

train_labels = []
for i in range(len(train_files)):
    train_labels.append(train_files[i])
    
valid_labels = []
for i in range(len(valid_files)):
    valid_labels.append(valid_files[i])
    
train_dataset = My_Dataset(train_files, train_labels, mode='train')
valid_dataset = My_Dataset(valid_files, valid_labels, mode='valid')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)   

# In[] Instantiation model
model = SRResNet(in_channels=3, out_channels=3)
# model = UNET(3, 64, 3)
# model = nn.DataParallel(model)    # parallel computing, open all the GPU
model = model.to(device)
summary(model, input_size=(3, 20000))   # 3 × 1 × 20000

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam optimizer

criterion = nn.MSELoss()
criterion.to(device)

# In[] Train & validation
model, early_stop, train_epochs_loss, valid_epochs_loss, epoch = train(model, 
                                                                       optimizer, 
                                                                       criterion, 
                                                                       train_dataloader, 
                                                                       valid_dataloader, 
                                                                       batch_size, 
                                                                       n_epochs, 
                                                                       device, 
                                                                       patience, 
                                                                       lr_decay)

# In[] Save loss & model
train_epochs_loss = np.array(train_epochs_loss)
valid_epochs_loss = np.array(valid_epochs_loss)

np.save(save_dir + 'train_epochs_' + str(epoch) + '_loss.npy', train_epochs_loss)
np.save(save_dir + 'valid_epochs_' + str(epoch) + '_loss.npy', valid_epochs_loss)
torch.save(model.state_dict(), save_dir + 'model_checkpoint_epoch_' + str(epoch) + '.pth')

# In[] Plot
epoch_plot = np.arange(1, len(train_epochs_loss) + 1)
plt.figure(figsize=(6, 4))
plt.plot(epoch_plot, train_epochs_loss, '-', label="train_loss")
plt.plot(epoch_plot, valid_epochs_loss, '-', label="valid_loss")
plt.title("Epochs Loss", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.savefig(fig_dir + "self_learning_loss.png", dpi=600, bbox_inches='tight')

# In[] Loading model infomation
model = SRResNet(in_channels=3, out_channels=3)
# model = UNET(3, 64, 3)
state_dict = torch.load(save_dir + 'model_checkpoint_epoch_' + str(epoch) + '.pth')
model.load_state_dict(state_dict)

# In[] Prediction & application
data = np.zeros((3, 20000))
data[0, :] = obspy.read(predict_data_dir + 'test/1.EHE.sac')[0].data
data[1, :] = obspy.read(predict_data_dir + 'test/1.EHN.sac')[0].data
data[2, :] = obspy.read(predict_data_dir + 'test/1.EHZ.sac')[0].data
target = np.zeros((3, 20000))
target[0, :] = obspy.read(predict_label_dir + 'test/1.EHE.sac')[0].data
target[1, :] = obspy.read(predict_label_dir + 'test/1.EHN.sac')[0].data
target[2, :] = obspy.read(predict_label_dir + 'test/1.EHZ.sac')[0].data

model = model.to(device)
for i in range(3):
    target[i, :] = max_min_normalization(target[i, :])
    data[i, :] = max_min_normalization(data[i, :])
data_tensor = (torch.from_numpy(data.astype(np.float32)).unsqueeze(0)).to(device)
output = (model(data_tensor).squeeze(0)).to('cpu')
output = (output.detach().numpy())
np.save(save_dir + 'pred_result.npy', output)