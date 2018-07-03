import numpy as np
import matplotlib.pyplot as plt

# Script that rotates the model and generates natural frequency plots
from bending import frequencies

angular_rate = 0.0
num_nodes = 4
num_freqs = 10
ref_speed = 109.12

# Generate fan plot
num_freqs = 4
Speeds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #, 1.1, 1.2, 1.3, 1.4, 1.5]
freqs = np.zeros((num_freqs, len(Speeds)))
snum = 0
for angular_rate in Speeds:
    omega = frequencies(angular_rate*ref_speed, num_nodes, num_freqs, ref_speed)
    print omega
    freqs[0:num_freqs,snum] = omega[0:num_freqs]
    snum += 1

for k in range(num_freqs):
    print Speeds,  freqs[k,:]
    plt.plot(Speeds, freqs[k,:])
plt.xlabel('$\Omega/\Omega_{ref}$')
plt.ylabel('$\omega/\Omega_{ref}$')
plt.grid()    
plt.savefig('fan.pdf', bbox_inches='tight', pad_inches=0.05)
