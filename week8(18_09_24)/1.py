import matplotlib.pyplot as plt
import math

# Function to perform convolution
def convolve(signal1, signal2):
    n = len(signal1)
    m = len(signal2)
    result_length = n + m - 1
    result = [0] * result_length
    
    for i in range(n):
        for j in range(m):
            result[i + j] += signal1[i] * signal2[j]
    
    return result

# Function to add two signals
def add_signals(signal1, signal2):
    max_len = max(len(signal1), len(signal2))
    result = [0] * max_len
    
    for i in range(max_len):
        if i < len(signal1):
            result[i] += signal1[i]
        if i < len(signal2):
            result[i] += signal2[i]
    
    return result

# Function to multiply two signals
def multiply_signals(signal1, signal2):
    min_len = min(len(signal1), len(signal2))
    result = [0] * min_len
    
    for i in range(min_len):
        result[i] = signal1[i] * signal2[i]
    
    return result

# Function to compute the spectrum (magnitude) of a signal
def compute_spectrum(signal):
    spectrum = [0] * len(signal)
    for i in range(len(signal)):
        spectrum[i] = abs(signal[i])  # Magnitude
    return spectrum

# Main program
def main():
    # Define a discrete signal
    discrete_signal = [1,2,3,4,5,6,7,8,9] # Example signal

    # Define an exponential signal
    exponential_signal = [math.exp(-0.5 * n) for n in range(len(discrete_signal))]

    # Perform convolution
    convoluted_signal = convolve(discrete_signal, exponential_signal)

    # Add signals
    added_signal = add_signals(discrete_signal, exponential_signal)

    # Multiply signals
    multiplied_signal = multiply_signals(discrete_signal, exponential_signal)

    # Compute spectra
    spectrum_convolution = compute_spectrum(convoluted_signal)
    spectrum_addition = compute_spectrum(added_signal)
    spectrum_multiplication = compute_spectrum(multiplied_signal)

    # Plotting the spectra
    plt.figure(figsize=(12,8))

    plt.subplot(3, 1, 1)
    plt.title('Spectrum of Convoluted Signal')
    plt.stem(spectrum_convolution, use_line_collection=True)

    plt.subplot(3, 1, 2)
    plt.title('Spectrum of Added Signal')
    plt.stem(spectrum_addition, use_line_collection=True)

    plt.subplot(3, 1, 3)
    plt.title('Spectrum of Multiplied Signal')
    plt.stem(spectrum_multiplication, use_line_collection=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
