import matplotlib.pyplot as plt
import numpy as np
import obspy
from config_param import Config

predict_label_dir = Config().predict_label_dir
predict_data_dir = Config().predict_data_dir
fig_dir = Config().fig_dir
save_dir = Config().save_dir

plt.figure(figsize=(15, 10))
xx = np.arange(0, 20, 0.001)

target = obspy.read(predict_label_dir + '230216.161712.EB000228/230216.161712.EB000228.EHZ.sac')[0].data
data = obspy.read(predict_data_dir + '230216.161712.EB000228/230216.161712.EB000228.EHZ.sac')[0].data
output = np.load(save_dir + 'pred_result.npy')[2, :]

plt.subplot(711)
plt.plot(xx, target, 'k')
plt.title('Noise-free signal', fontsize=14)

plt.subplot(712)
plt.plot(xx, data - target, 'k')
plt.title('Noise', fontsize=14)
plt.ylim(-1, 1)

plt.subplot(713)
plt.plot(xx, data, 'k')
plt.title('Noisy data', fontsize=14)

plt.subplot(714)
plt.plot(xx, output, 'k')
plt.title('Denoised signal', fontsize=14)

plt.subplot(715)
plt.plot(xx, target - output, 'k')
plt.title('Noise-free signal - Denoised signal', fontsize=14)
plt.ylim(-1, 1)

plt.subplot(716)
plt.plot(xx, data - output, 'k')
plt.title('Removed noise', fontsize=14)
plt.ylim(-1, 1)

plt.subplot(717)
plt.plot(xx, (data - target) - (data - output), 'k')
plt.title('Noise - Removed noise', fontsize=14)
plt.ylim(-1, 1)

plt.tight_layout()

plt.savefig(fig_dir + 'Prediction.png', dpi=600, bbox_inches='tight')