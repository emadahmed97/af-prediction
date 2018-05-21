import sys 
import numpy as np

from matplotlib import pyplot as plt

sys.path.append('..')
import cardio
import cardio.dataset as ds
index = ds.FilesIndex(path='C:/training2017/A*.hea', no_ext=True, sort=True)

#print(index.indices)

from cardio import EcgBatch
eds = ds.Dataset(index, batch_class=EcgBatch)
batch = eds.next_batch(batch_size=6, unique_labels=['A', 'N', 'O'])
batch_with_data = batch.load(fmt='wfdb', components=['signal', 'meta'])
batch_with_data = batch.load(src='C:/training2017/REFERENCE.csv', fmt='csv', components='target')
#print(batch_with_data['A00001'].target)

## now trying different actions
original_batch = batch_with_data.deepcopy()
changed_batch = batch_with_data.deepcopy()

## CONVOLUTION to remove noise from Ecg
kernel = cardio.kernels.gaussian(size=10) # will convolute signal with gaussian curve to remove noise

length = original_batch["A00001"].signal.shape[1]

noise = np.random.normal(scale=0.1, size=length)

original_batch["A00001"].signal += noise
changed_batch["A00001"].signal += noise

changed_batch.convolve_signals(kernel=kernel)


original_batch.show_ecg('A00001', start=10, end=15) # noisy

changed_batch.show_ecg('A00001', start=10, end=15) # noise filtered out 

# BANDPASS - Continue later

