import numpy as np
import matplotlib.pyplot as plt


Fs = 8000  
T = 1      
f = 200   

t = np.arange(0, T, 1/Fs)
x = np.sin(2 * np.pi * f * t)


X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), 1/Fs)


idx = np.argmax(np.abs(X))
dominant_frequency = freqs[idx]


plt.plot(t, x)
plt.title(f'Sinusoidal Wave: {f} Hz')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
