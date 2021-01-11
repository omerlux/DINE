import numpy as np
import matplotlib.pyplot as plt
import re

valid_f1 = []
valid_f2 = []
valid_loss = []
train_f1 = []
train_f2 = []
train_loss = []

inp = input("Insert relative path from ./dine/ folder: ")
file = open("./dine/" + inp + "/log.txt")
P = input("Insert P value: ")
N = input("Insert N value: ")
db = float(input("Insert dB SNR value (-999 to see P and N): "))
dim = int(input("Insert vector dimension: "))
process = input("Insert process type: ")
capacity = float(input("Insert capacity value (0 for no capacity line): "))

valid_start = 0
for line in file:
    numbers = re.findall("[^a-zA-Z:](\-?\d+[\.]?\d*)", line)
    if "valid loss" in line:
        if not valid_start:
            valid_start = len(train_f1)
        valid_f1.append(float(numbers[-3]))
        valid_f2.append(float(numbers[-2]))
        valid_loss.append(float(numbers[-1]))
    elif "batches" in line:
        train_f1.append(float(numbers[-3]))
        train_f2.append(float(numbers[-2]))
        train_loss.append(float(numbers[-1]))

epoch_train = np.arange(0, len(train_f1), 1) / valid_start
epoch_valid = np.arange(1, len(valid_f1)+1, 1)

plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(epoch_train, train_loss, linewidth=1, color='blue', label='training')
plt.plot(epoch_valid, valid_loss, linewidth=1, color='red', label='validation')
if capacity:
    plt.plot(epoch_train, [capacity]*len(train_loss), linewidth=1, color='green', label='capacity')
plt.grid(True, which='both', axis='both')
if db != -999:
    plt.title('DINE - Capcity of {} - dim={}, SNR {} dB'.format(process, dim, db))
else:
    plt.title('DINE - Capcity of {} - dim={}, P={}, N={}'.format(process, dim, P,N))
plt.xlabel('Epochs')
plt.ylabel('Capacity')
plt.legend()
plt.savefig("./dine/" + inp +"/capacity.png", dpi=1200)
plt.show()

plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(epoch_train, train_f1, linewidth=1, color='blue', label='training F1')
plt.plot(epoch_valid, valid_f1, linewidth=1, color='red', label='validation F1')
plt.grid(True, which='both', axis='both')
if db != -999:
    plt.title('DINE - F1 of {} - dim={}, SNR {} dB'.format(process, dim, db))
else:
    plt.title('DINE - F1 of {} - dim={}, P={}, N={}'.format(process, dim, P,N))
plt.xlabel('Epochs')
plt.ylabel('F1 Value')
plt.legend()
plt.savefig("./dine/" + inp +"/f1 convergence.png", dpi=1200)
plt.show()

plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(epoch_train, train_f2, linewidth=1, color='blue', label='training F2')
plt.plot(epoch_valid, valid_f2, linewidth=1, color='red', label='validation F2')
plt.grid(True, which='both', axis='both')
if db != -999:
    plt.title('DINE - F2 of {} - dim={}, SNR {} dB'.format(process, dim, db))
else:
    plt.title('DINE - F2 of {} - dim={}, P={}, N={}'.format(process, dim, P,N))
plt.xlabel('Epochs')
plt.ylabel('F1 Value')
plt.legend()
plt.savefig("./dine/" + inp +"/f2 convergence.png", dpi=1200)
plt.show()