import numpy as np
import matplotlib.pyplot as plt
z=int(input("Zero location in Z-plane: "))
r=int(input("Zero location from origin: "))
wo=float(input("Freq of Zeroes: "))
z1=r*np.exp(1j*wo)
w=np.arange(-1*np.pi,np.pi,0.01)
H=1-z1*(np.exp(-1j*w))
plt.plot(w,np.abs(H),label='Magnitude')
plt.plot(w,np.angle(H),label='Phase')
plt.grid(True)
plt.legend()
plt.show()