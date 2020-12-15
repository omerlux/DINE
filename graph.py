import numpy as np
import matplotlib.pyplot as plt
import re

db = range(-15, 15+1, 5)
test_cases = [0.03, 0.07, 0.16, 0.39, 0.77, 1.24, 1.81]

db2 = np.arange(-20, 15+0.25, 0.25)
capacity = [0.5 * np.log(1 + 10 ** (x / 10)) for x in db2]

plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(db, test_cases, color='blue', linestyle='None', marker='o', label='test capacity')
plt.plot(db2, capacity, linewidth=1, color='red', label='ground truth capacity')
plt.grid(True, which='both', axis='both')
plt.title('DINE - Capcity of AWGN - dim=1')
plt.xlabel('P/\u03C3\u00b2 (SNR in dB)')
plt.ylabel('Capacity')
plt.legend()
plt.savefig("./dine/capacity dim=1.png", dpi=1200)
plt.show()
