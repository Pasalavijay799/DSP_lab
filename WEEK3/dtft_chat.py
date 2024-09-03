import numpy as np
import matplotlib.pyplot as plt

def dtft(x_n, omega):

    n = np.arange(len(x_n))  

    #np.meshgrid(n, omega, indexing='ij') creates coordinate matrices from the coordinate vectors n and omega. This enables broadcasting for element-wise operations.
    n, omega = np.meshgrid(n, omega, indexing='ij')
    #axis=0 is used to sum across the time dimension to compute the DTFT for each frequency
    X_w = np.sum(x_n[n] * np.exp(-1j * omega * n),axis=0)
    return X_w

def plot_dtft(x_n):
    """
    Plot the DTFT of the signal x[n] in terms of magnitude and phase.

    Parameters:
        x_n (array-like): The discrete-time signal.
    """
    omega = np.arange(-np.pi, np.pi, 0.0001 * np.pi)  # Frequency range from -π to π
    X_w = dtft(x_n, omega)

    # Plot Magnitude and Phase
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(omega, np.abs(X_w))
    plt.title('Magnitude of DTFT')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('|X(ω)|')

    plt.subplot(2, 1, 2)
    plt.plot(omega, np.angle(X_w))
    plt.title('Phase of DTFT')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('∠X(ω)')
#plt.tight_layout() adjusts subplot parameters to give some padding and prevent overlapping.
    plt.tight_layout()
    plt.show()

# Example usage
x_n = np.array([1, 2, 3, 4])  # Example signal
plot_dtft(x_n)
