import wfdb
import sys
import numpy as np

from matplotlib import pyplot

sys.path.append('..')

import cardio.dataset as ds
from cardio import EcgBatch

import cardio
import matplotlib.pyplot as plt

# Working With Files

index = ds.FilesIndex(path='cardio/tests/data/training2017/A*.hea', no_ext=True, sort=True)

print(index.indices)

eds = ds.Dataset(index, batch_class=EcgBatch)

batch = eds.next_batch(batch_size=6, unique_labels=['A', 'N', 'O'])

batch_with_data = batch.load(fmt='wfdb', components=['signal', 'meta'])

# batch_with_data.show_ecg('A00001')

# CONVOLUTION:
kernel = cardio.kernels.gaussian(size=11)

plt.plot(kernel)
plt.grid("on")
# plt.show()


original_batch = batch_with_data.deepcopy()
changed_batch = batch_with_data.deepcopy()

siglen = original_batch["A00001"].signal.shape[1]

noise = np.random.normal(scale=0.01, size=siglen)

original_batch["A00001"].signal += noise
changed_batch["A00001"].signal += noise

changed_batch.convolve_signals(kernel=kernel)

# original_batch.show_ecg('A00001', start=10, end=15)

# changed_batch.show_ecg('A00001', start=10, end=15)

# Bandpass
original_batch = batch_with_data.deepcopy()
changed_batch = batch_with_data.deepcopy()

n_sin = 10
noise = np.zeros_like(original_batch['A00001'].signal)
siglen = original_batch['A00001'].signal.shape[1]
t = np.linspace(0, 30, siglen)
for i in range(n_sin):
    a = np.random.uniform(0, 0.1)
    omega = np.random.uniform(0.1, 0.8)
    phi = np.random.uniform(0, 2 * np.pi)
    noise += a * np.sin(omega * t + phi)

fig = plt.figure(figsize=(12, 4))
plt.plot(noise[0])
plt.grid("on")
#plt.show()

original_batch['A00001'].signal += noise
changed_batch['A00001'].signal += noise

changed_batch.band_pass_signals(low=0.2)

original_batch.show_ecg('A00001')
changed_batch.show_ecg('A00001')
