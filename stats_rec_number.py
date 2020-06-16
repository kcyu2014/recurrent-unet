import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def get_miou(file):
    text_file = open(file, "r")
    lines = text_file.read().split('\n')
    text_file.close()
    miou = []
    for line in lines:
        if 'IoU' in line:
            # print(line)
            name, score = line.split('\t')
            miou.append(float(score))
    n = int(len(miou)/2)
    miou_val = np.asarray(miou[0:n])
    miou_test = np.asarray(miou[n::])

    return miou_val, miou_test


run_dirs = [
    'eyth_hand/iccvablation/dru_eythhand-h128-1-r12-w-0.4-gate3-bs-8-fscale-4/76229',
    'eyth_hand/iccvablation/dru_eythhand-h128-1-r9-w-0.4-gate3-bs-8-fscale-4/94411',
    'eyth_hand/iccvablation/dru_eythhand-h128-1-r6-w-0.4-gate3-bs-8-fscale-4/51149',
    'eyth_hand/iccvablation/dru_eythhand-h128-1-r3-w-0.4-gate3-bs-8-fscale-4/33301']

eval_files = [
    'eval_2019_02_27_11_49_42.log',
    'eval_2019_02_27_11_50_37.log',
    'eval_2019_02_27_11_51_28.log',
    'eval_2019_02_27_11_52_20.log']

list_miou_val = []
list_miou_test = []
for i, j in zip(run_dirs, eval_files):
    file = os.path.join('runs', i, j)
    miou_val, miou_test = get_miou(file)
    list_miou_val.append(miou_val)
    list_miou_test.append(miou_test)

# list_miou = list_miou_val
list_miou = list_miou_test

t = np.arange(1, len(list_miou_val[0])+1)

fig, ax = plt.subplots()

plt.plot(t, list_miou[0], 'ro-', label='Train-Rec-12')
plt.plot(t, list_miou[1], 'bo-', label='Train-Rec-9')
plt.plot(t, list_miou[2], 'go-', label='Train-Rec-6')
plt.plot(t, list_miou[3], 'yo-', label='Train-Rec-3')

plt.ylim(0.83, 0.87)
plt.xlabel('Test-Rec-Number', fontsize=16)
plt.ylabel('mIoU', fontsize=16)

ax.set_xticks(t)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(loc='lower right', prop={'size': 16})

# plt.show()
plt.savefig('runs/eyth_hand/iccvablation/eyth_test_dru.pdf', bbox_inches='tight')
print('plot done!')
