import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Parameters
L = 1.0                # Length of the string (meters)
T = 100.0              # Tension in the string (Newtons) 100.0
rho = 0.01             # Linear density of the string (kg/m) 0.01
Y = 2e11               # Young's modulus (Pascals) 2e11
r = 0.001              # Radius of the string (meters) 0.001
A_string = np.pi * r**2 # Cross-sectional area (m^2)
K = r / 2              # Radius of gyration for cylindrical string (meters)
K_squared = K**2       # K squared (m^2)

# Wave speed
c = np.sqrt(T / rho)   # Wave speed (m/s)

# Discretization parameters
dx = 0.01              # Spatial step size (meters)
dt = 0.00001        # Time step size (seconds) - reduced for stability 0.00001
t_max = 2.0            # Total time to simulate (seconds)

# Stability condition
if dt >= dx / c:
    raise ValueError("Time step size dt is too large for stability condition.")

# Derived parameters
N = int(L / dx) + 1    # Number of spatial steps
M = int(t_max / dt)    # Number of time steps
x = np.linspace(0, L, N)

# Different stiffness values
stiffness_values = [1e9, 5e10, 1e11, 2e11]
#stiffness_values = [1e2,1e3,1e4,1e5]

for idx, Y in enumerate(stiffness_values):
    print("Y",Y)
    y = np.zeros(N)
    y_new = np.zeros(N)
    y_old = np.zeros(N)

    # Initial conditions (a pluck at the center of the string)
    y[int(N/2)] = 0.1

    frames = 5

    # Finite difference method
    for t in range(M):
        n = 1
        for i in range(2, N-2):
            term1 = 2 * y[i] - y_old[i]
            term2 = (c * dt / dx) ** 2 * (y[i+1] - 2 * y[i] + y[i-1])
            term3 = (K_squared * Y * dt ** 2 / (rho * dx ** 4)) * (y[i-2] - 4 * y[i-1] + 6 * y[i] - 4 * y[i+1] + y[i+2])
            y_new[i] = term1 + term2 - term3

        # Apply boundary conditions (fixed ends)
        y_new[0] = 0
        y_new[1] = 0
        y_new[-1] = 0
        y_new[-2] = 0

        # Update old and current arrays
        y_old[:] = y
        y[:] = y_new

        # if t % (n*M/frames) != 0:    
        #     plt.plot(x,y)
        #     plt.show()
        #     n+=1

    # Plot the final state of the string
    plt.plot(x, y, label=f'Final State (Y = {Y} Pa)')
    plt.xlabel('Position along the string (m)')
    plt.ylabel('Displacement (m)')
    plt.title(f'String Vibration with Stiffness Y = {Y} Pa')
    plt.legend()
    plt.show()

    print(y)
    # Save the solution as a sound file
    sampling_rate = 44100
    waveform = np.interp(np.linspace(0, N, M), np.arange(N), y)
    waveform = np.int16((waveform / np.max(np.abs(waveform))) * 32767)
    write(f'string_vibration_Y_{Y}.wav', sampling_rate, waveform)

# Visualize how stiffness depends on length, area, and Young's modulus
lengths = np.linspace(0.5, 2.0, 100)
areas = np.pi * (np.linspace(0.0005, 0.005, 100)**2)
young_moduli = np.linspace(1e11, 3e11, 100)

stiffness_length = K_squared * Y / lengths
stiffness_area = K_squared * Y / areas
stiffness_ymodulus = K_squared * young_moduli / A_string

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(lengths, stiffness_length)
plt.xlabel('Length (m)')
plt.ylabel('Stiffness (N/m^2)')
plt.title('Stiffness vs Length')

plt.subplot(1, 3, 2)
plt.plot(areas, stiffness_area)
plt.xlabel('Area (m^2)')
plt.ylabel('Stiffness (N/m^2)')
plt.title('Stiffness vs Area')

plt.subplot(1, 3, 3)
plt.plot(young_moduli, stiffness_ymodulus)
plt.xlabel('Young\'s Modulus (Pa)')
plt.ylabel('Stiffness (N/m^2)')
plt.title('Stiffness vs Young\'s Modulus')

plt.tight_layout()
plt.show()