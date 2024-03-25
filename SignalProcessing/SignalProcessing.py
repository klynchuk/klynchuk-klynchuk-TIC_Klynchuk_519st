import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

def draw_graph(data_y, data_x, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(data_x, data_y, linewidth=1)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=15)
    fig.savefig('./figures/'+title+'.png', dpi=600)

n_points = 500
random_data = np.random.normal(0, 15, n_points)
sampling_freq = 1000

max_frequency = 17
nyquist_freq = sampling_freq / 2
cutoff_freq = max_frequency / nyquist_freq

filter_parameters = signal.butter(3, cutoff_freq, 'low', output='sos')

time = np.arange(n_points) / sampling_freq
filtered_data = signal.sosfiltfilt(filter_parameters, random_data)
draw_graph(filtered_data, time, 'Спектр сигналу з максимальною частотою F_max = 17 Гц', 'Частота (Гц)', 'Амплітуда спектру')
plt.show()

spectrum = fft.fft(filtered_data)
spectrum = np.abs(fft.fftshift(spectrum))
freq = fft.fftfreq(n_points, 1/n_points)
freq = fft.fftshift(freq)
draw_graph(spectrum, freq, 'Сигнал з максимальною частотою F_max = 17 Гц', 'Час, (секунди)', 'Амплітуда сигналу')
plt.show()
