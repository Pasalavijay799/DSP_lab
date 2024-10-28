import numpy as np

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def overlap_save(x, h, M):
    L = len(h)               
    N = M + L - 1            
    P = L - 1                
    
  
    h_padded = np.zeros(N)
    h_padded[:L] = h
    

    H = dft(h_padded)
   
    y = []

    for i in range(0, len(x), M):
        if i == 0:
            x_segment = np.zeros(N)
            x_segment[-M:] = x[:M]
        else:
            x_segment = np.zeros(N)
            x_segment[:P] = x[i-P:i]
            x_segment[P:] = x[i:i+M]
        
        X = dft(x_segment)

        Y = X * H
        
        y_segment = np.real(idft(Y))
  
        if i == 0:
            y.extend(y_segment[P:]) 
        else:
            y.extend(y_segment[P:M+P])  

    return np.array(y)
55
x = np.array([3, -1, 0, 3, 2, 0, 1, 2, 1])
h = np.array([1, 1, 1])
M = 3
y = overlap_save(x, h, M)
print("Output:", y)
