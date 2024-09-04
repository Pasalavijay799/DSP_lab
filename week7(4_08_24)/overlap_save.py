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

def overlap_save(x, h):
    L = len(h)               # Length of the filter
    N = 2 * L - 1            # Segment size, typically L + M - 1
    M = len(x)               # Length of the input signal
    
    # Zero-pad the filter to length N
    h_padded = np.zeros(N)
    h_padded[:L] = h
    
    # Compute the DFT of the filter
    H = dft(h_padded)
    
    # Initialize the output array
    y = np.zeros(M + L - 1)
    
    # Process each segment
    for i in range(0, M, L):
        # Extract the current segment with zero-padding
        x_segment = np.zeros(N)
        if i + L <= M:
            x_segment[:L] = x[i:i+L]
        else:
            x_segment[:M-i] = x[i:]
        
        # Compute the DFT of the segment
        X = dft(x_segment)
        
        # Multiply in the frequency domain
        Y = X * H
        
        # Compute the inverse DFT
        y_segment = np.real(idft(Y))
        
        # Overlap-save: discard the first L-1 points and add the rest to the output
        if i == 0:
            y[i:i+L] = y_segment[L-1:]  # No overlap on the first segment
        else:
            y[i:i+L] += y_segment[L-1:]
    
    # Return the valid part of the output
    return y[:M + L - 1]

# Example usage
x = np.array([3, -1, 0, 1, 3, 2, 0, 1, 2, 1])
h = np.array([1, 1, 1])
y = overlap_save(x, h)
print("Output:", y)
