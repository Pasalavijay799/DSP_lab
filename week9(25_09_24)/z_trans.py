import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the variable z for the Z-transform
z = sp.symbols('z', complex=True)
n = sp.symbols('n', integer=True)

def compute_z_transform(sequence, n_var, L):
    
    
    # Compute the Z-transform as a summation
    Z = sp.summation(sequence * z**(-n_var), (n_var, 0, L))
    
    return sp.simplify(Z)  # Simplify the result for a cleaner output

       
def plot_zeros_poles(zeros,poles):
    
    
    # Convert zeros and poles to numpy arrays for plotting
    zeros = np.array([complex(zero) for zero in zeros], dtype=np.complex128)
    poles = np.array([complex(pole) for pole in poles], dtype=np.complex128)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)

    # Plot zeros
    plt.scatter(zeros.real, zeros.imag, s=100, c='b', marker='o', label='Zeros')
    
    # Plot poles
    plt.scatter(poles.real, poles.imag, s=100, c='r', marker='x', label='Poles')
    
    # Add unit circle for stability reference
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_artist(unit_circle)

    plt.title('Zeros and Poles in the Z-Plane')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
    

# Example usage

L = sp.oo # infinity
sequence = 1  

z_transform = compute_z_transform(sequence, n, L)
print("Z-transform:", z_transform)
numerator, denominator = z_transform.args[0][0].as_numer_denom()
print('Numerator: ',numerator,' Denominator: ', denominator)


numerator = z**2 + 4*z + 8  #  z^2 + 2z + 1
denominator = z - 1         #  z - 1

# transfer function
H_z = numerator / denominator
print("H(z):", H_z)

# Find the roots of the denominator (poles)
poles = sp.solve(denominator, z)
print("Poles of the system:", poles)

# Find the roots of the numerator (zeros)
zeros = sp.solve(numerator, z)
print("Zeros of the system:", zeros)

plot_zeros_poles(zeros,poles)



