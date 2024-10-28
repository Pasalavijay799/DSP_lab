import numpy as np

def circular_convolution(x, h):
    N = max(len(x), len(h))
    x_padded = np.pad(x, (0, N - len(x)))
    h_padded = np.pad(h, (0, N - len(h)))
    print("x:",x_padded)
    print("h:",h_padded)
    y = np.zeros(N)
    
    for n in range(N):
        sum = 0
        for k in range(N):
            sum += x_padded[k] * h_padded[(n - k) % N]
        y[n] = sum
    
    return y

x = [4,3,2,1]
h = [1,-2,0,4]

result = circular_convolution(x, h)
print("Circular Convolution Result:",[int(i) for i in  result])
