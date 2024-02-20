import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('TkAgg')
import numpy as np
from numpy.fft import fft
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24}

# enrollment time against EER
# 5s 10s 15s 20s 25s
enroll_time = [5, 10, 15, 20, 25]
EERs = np.array([0.0254, 0.0196, 0.0176, 0.0174, 0.0166]) * 100
plt.figure()
plt.plot(enroll_time, EERs)
plt.xlabel('Time (s)', fontdict=font1)
plt.ylabel('EER (%)', fontdict=font1)
plt.savefig('eer_time.pdf')
