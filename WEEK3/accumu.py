import numpy as np
import matplotlib.pyplot as plt

# Example signal: a simple array of values
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Compute the cumulative sum (accumulation)
cumulative_sum = np.cumsum(signal)

# Print the results
print("Original Signal:", signal)
print("Cumulative Sum:", cumulative_sum)

# Optional: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original Signal', marker='o')
plt.plot(cumulative_sum, label='Cumulative Sum', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Signal and its Cumulative Sum')
plt.legend()
plt.grid(True)
plt.show()
