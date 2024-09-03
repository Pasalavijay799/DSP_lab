import numpy as np
import matplotlib.pyplot as plt


def dtft(x):
    X = []
    w = np.arange(-np.pi, np.pi, 0.0001*np.pi)
    for f in w:
        s = 0
        for n in range(len(x)):
            s += x[n] * np.exp(-1j * f * n)
        X.append(s)
    return w, np.array(X)

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])


w, X1 = dtft(x1)
_, X2 = dtft(x2)

X1_plus_X2 = X1 + X2
_, X3 = dtft(x1 + x2)

if np.allclose(X1_plus_X2, X3):
    print("X1 + X2 == X3 condition satisfied")

    d = 3
    X1_freq_shifted = dtft(x1 - d)[1] 
    X1_freq_shifted_theoretical = X1 * np.exp(-1j * d * w)
    
    
    x1_time_delayed = np.concatenate([x1, np.zeros(len(x1))])
    x2_time_delayed = np.concatenate([np.zeros(len(x2)), x2])
    _, X1_time_delayed = dtft(x1_time_delayed)
    _, X2_time_delayed = dtft(x2_time_delayed)
    
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(w, np.abs(X1), label='X1')
    plt.title('DTFT of x1')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.subplot(3, 2, 2)
    plt.plot(w, np.abs(X2), label='X2')
    plt.title('DTFT of x2')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.subplot(3, 2, 3)
    plt.plot(w, np.abs(X1_plus_X2), label='X1 + X2')
    plt.title('DTFT of X1 + X2')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.subplot(3, 2, 4)
    plt.plot(w, np.abs(X3), label='X3')
    plt.title('DTFT of x1 + x2')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.subplot(3, 2, 5)
    plt.plot(w, np.abs(X1_freq_shifted_theoretical), label='Frequency Shifted X1')
    plt.title('Frequency Shifted DTFT of x1')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.subplot(3, 2, 6)
    plt.plot(w, np.abs(X1_time_delayed), label='Time Delayed X1')
    plt.title('Time Delayed DTFT of x1')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
else:
    print("X1 + X2 != X3 condition not satisfied")
