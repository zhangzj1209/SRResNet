import numpy as np

def gen_gauss_noise(signal, SNR):
    """
    :param signal: raw signal
    :param SNR: signal-to-noise ratio of the added noise
    :return: generated noise
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape -> bbtain the size of the sample
    noise = noise - np.mean(noise)
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal, 2))
    noise_variance = signal_power/np.power(10, (SNR/10))
    noise = (np.sqrt(noise_variance)/np.std(noise))*noise
    return noise

def check_SNR(signal, noise):
    '''
    :param signal: raw signal
    :param noise: generated noise
    :return: signal-to-noise ratio
    '''
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal, 2))
    noise_power = (1/noise.shape[0])*np.sum(np.power(noise, 2))
    SNR = 10*np.log10(signal_power/noise_power)
    return SNR