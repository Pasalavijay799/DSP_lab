import numpy as np
from matplotlib import pyplot as plt

# Get zero location from user
z = input("Enter the zero location (in form 'x+yj'): ")
zero = complex(z)

# Get radius and frequency from user
r = float(input("Enter the distance from the origin (r): "))
w0 = float(input("Enter the frequency of the zero (w0): "))

# Calculate z1 and h
z1 = r * np.exp(1j * w0)  # Use 1j instead of np.j
h = 1 / (1 - z1 * np.exp(1j * w0))

# Prepare for plotting frequency response
x = np.linspace(-10, 10, 100)  # Adjust range as needed
h_values = [1 / (1 - z1 * np.exp(1j * w)) for w in x]

# Plot the magnitude and angle of h
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x, np.abs(h_values), label='Magnitude')
plt.title('Magnitude of h')
plt.xlabel('Frequency (w)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, np.angle(h_values), label='Angle', color='orange')
plt.title('Angle of h')
plt.xlabel('Frequency (w)')
plt.ylabel('Angle (radians)')
plt.grid()
plt.legend()

plt.tight_layout()

# Plot the z-plane (zero and poles)
plt.figure(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 300)

# Plot the unit circle
plt.plot(np.cos(theta), np.sin(theta), linestyle='--', label='Unit Circle')

# Plot the zero location
plt.scatter(np.real(zero), np.imag(zero), color='red', s=100, label='Zero', zorder=5)

# Plot the pole location (since z1 is the pole here)
pole = z1
plt.scatter(np.real(pole), np.imag(pole), color='blue', s=100, label='Pole', zorder=5)

# Labeling the plot
plt.title('Z-Plane Plot')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()

plt.show()

