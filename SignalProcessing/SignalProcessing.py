import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

def draw_graph(data_y, data_x, title, x_label, y_label, ylim=None):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(data_x, data_y, linewidth=1)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.title(title, fontsize=15)
    fig.savefig('./figures/'+title+'.png', dpi=600)

# Задані параметри
n_points = 500
random_data = np.random.uniform(-4, 4, n_points)
sampling_freq = 1000
max_frequency = 17
nyquist_freq = sampling_freq / 2
F_filter = 24

# Фільтрація сигналу
cutoff_freq = max_frequency / nyquist_freq
filter_parameters = signal.butter(3, cutoff_freq, 'low', output='sos')
time = np.arange(n_points) / sampling_freq
filtered_data = signal.sosfiltfilt(filter_parameters, random_data)

# Дискретизація сигналу та обчислення дисперсії
step_sizes = [2, 4, 8, 16]
variance_diff = []
snr = []

for Dt in step_sizes:
    discrete_signal = np.zeros(n_points)
    for i in range(0, round(n_points/Dt)):
        discrete_signal[i * Dt] = filtered_data[i * Dt]
    restored_signal = signal.sosfiltfilt(filter_parameters, discrete_signal)
    difference = restored_signal - random_data
    variance_diff.append(np.var(difference))
    snr.append(np.var(random_data) / np.var(difference))

# Побудова графіка залежності дисперсії від кроку дискретизації
draw_graph(variance_diff, step_sizes, 'Залежність дисперсії від кроку дискретизації', 'Крок дискретизації', 'Дисперсія')

# Побудова графіка залежності сигнал-шум від кроку дискретизації
draw_graph(snr, step_sizes, 'Залежність сигнал-шум від кроку дискретизації', 'Крок дискретизації', 'Співвідношення сигнал-шум')

# Побудова графіків дискретизованих сигналів з кривими залежностями
discrete_signals = []
discrete_curves = []
for Dt in step_sizes:
    discrete_signal = np.zeros(n_points)
    discrete_curve = np.zeros(n_points)
    for i in range(0, round(n_points/Dt)):
        discrete_signal[i * Dt] = filtered_data[i * Dt]
    discrete_signals.append(list(discrete_signal))
    discrete_curve[:round(n_points/Dt)*Dt:Dt] = filtered_data[:round(n_points/Dt)*Dt:Dt]
    discrete_curves.append(list(discrete_curve))

# Побудова графіків дискретизованих сигналів з кривими залежностями
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time, discrete_curves[s], linewidth=1)
        ax[i][j].set_xlabel('Час (секунди)', fontsize=15)
        ax[i][j].set_ylabel('Амплітуда сигналу', fontsize=15)
        ax[i][j].set_ylim(-4, 4)
        s += 1
fig.suptitle('Дискретизовані сигнали', fontsize=15)
fig.savefig('./figures/Дискретизовані_сигнали.png', dpi=600)

# Відновлення аналогового сигналу з дискретного
restored_signals = []
for discrete_signal in discrete_signals:
    restored_signal = signal.sosfiltfilt(filter_parameters, discrete_signal)
    restored_signals.append(restored_signal)

# Побудова графіків відновлених аналогових сигналів
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(time, restored_signals[s], linewidth=1)
        ax[i][j].set_xlabel('Час (секунди)', fontsize=15)
        ax[i][j].set_ylabel('Амплітуда сигналу', fontsize=15)
        s += 1
fig.suptitle('Відновлені аналогові сигнали', fontsize=15)
fig.savefig('./figures/Відновлені_аналогові_сигнали.png', dpi=600)

# Розрахунок та побудова спектрів сигналів
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        spectrum = fft.fft(discrete_signals[s])
        spectrum = np.abs(fft.fftshift(spectrum))
        freq = fft.fftfreq(len(discrete_signals[s]), 1/n_points)
        freq = fft.fftshift(freq)
        if s == 0:
            ax[i][j].set_xlim(-200, 200)
            ax[i][j].set_ylim(0, 150)
        elif s == 1:
            ax[i][j].set_xlim(-200, 200)
            ax[i][j].set_ylim(0, 80)
        elif s == 2:
            ax[i][j].set_xlim(-200, 200)
            ax[i][j].set_ylim(0, 40)
        elif s == 3:
            ax[i][j].set_xlim(-200, 200)
            ax[i][j].set_ylim(0, 20)
        ax[i][j].plot(freq, spectrum, linewidth=1)
        ax[i][j].set_xlabel('Частота (Гц)', fontsize=15)
        ax[i][j].set_ylabel('Амплітуда спектра', fontsize=15)
        s += 1
fig.suptitle('Спектри дискретизованих сигналів', fontsize=15)

# Побудова графіків спектра сигналу з максимальною частотою F_max = 17 Гц
time = np.arange(n_points) / sampling_freq
filtered_data = signal.sosfiltfilt(filter_parameters, random_data)
draw_graph(filtered_data, time, 'Спектр сигналу з максимальною частотою F_max = 17 Гц', 'Частота (Гц)', 'Амплітуда спектру')

# Розрахунок та побудова спектру сигналу
spectrum = fft.fft(filtered_data)
spectrum = np.abs(fft.fftshift(spectrum))
freq = fft.fftfreq(n_points, 1/n_points)
freq = fft.fftshift(freq)
draw_graph(spectrum, freq, 'Сигнал з максимальною частотою F_max = 17 Гц', 'Час, (секунди)', 'Амплітуда сигналу')

fig.savefig('./figures/Спектри_дискретизованих_сигналів.png', dpi=600)
# Показати графіки
plt.show()

