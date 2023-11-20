import re
import matplotlib.pyplot as plt

loss_values = []
reg_values = []

with open('Loss_log_attention.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        match_loss = re.search(r'Loss: ([\-\d.]+)', line)
        match_reg = re.search(r'Reg: ([\-\d.]+)', line)
        if match_loss and match_reg:
            loss_values.append(float(match_loss.group(1)))
            reg_values.append(float(match_reg.group(1)))

epochs = range(len(loss_values))
plt.xlabel('Step')
plt.ylabel('Loss')
plt.plot(epochs, loss_values, label='Loss')
plt.legend()
plt.savefig('loss.png')

plt.clf()
plt.xlabel('Step')
plt.ylabel('Reg')
plt.plot(epochs, reg_values, label='Reg')
plt.legend()
plt.savefig('Reg.png')