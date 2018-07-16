import numpy as np
import matplotlib.pyplot as plt

# Script that rotates the model and generates natural frequency plots
from test_beam import frequencies

num_freqs = 10
num_nodes = 2
ref_speed = 109.12

file = open("freqdata.dat", "w") 

# Generate fan plot
Speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
freqs = np.zeros((num_freqs, len(Speeds)))
snum = 0
for angular_rate in Speeds:
    omega = frequencies(angular_rate*ref_speed, num_nodes, num_freqs, ref_speed)
    freqs[0:num_freqs,snum] = omega[0:num_freqs]
    snum += 1

    # write to file
    file.write(('Omega = %e\n') % (angular_rate*ref_speed))
    for f in omega[0:num_freqs]:
        file.write(('%e\n')%(f))

# write to file
file.close()
