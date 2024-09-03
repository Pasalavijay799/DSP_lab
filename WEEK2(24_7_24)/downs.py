

def downsampling(x, a):
    if a > 1:
        y = []
        for i in range(0, len(x)):
            if i % a == 0: 
                y.append(x[i])
        return y

def upsampling(x, b):
    if b < 1:
        l = []
        for i in range(len(x)):
            for j in range(int(1/b)):
                l.append(x[i])
        return l

x = [1, 2, 3, 4, 5]

z = downsampling(x, 3)
k = upsampling(x, 1/3)

print("Downsampled:", z)
print("Upsampled:", k)
