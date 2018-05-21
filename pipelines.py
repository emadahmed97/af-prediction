import sys
import os
sys.path.append("..")

import cardio.dataset as ds
from cardio.dataset import B
from cardio import EcgDataset

filter_pipeline = (ds.Pipeline()
                    .load(fmt="wfdb", components=["signal", "meta"])
                    .band_pass_signals(low=5, high=40)
                    )
PATH_TO_DATA = "C:/training2017"
eds = EcgDataset(path=os.path.join(PATH_TO_DATA, "A*.hea"), no_ext=True, sort=True)
(eds >> filter_pipeline).run(batch_size=len(eds), n_epochs=1)

## at this point the filtered ech is gone, batches are destroyed within Pipeline
## use pipeline variables to store data that will be used later

filter_pipeline = (ds.Pipeline()
                     .init_variable('saved_batches', init_on_each_run=list)
                     .load(fmt="wfdb", components=["signal", "meta"])
                     .update_variable('saved_batches', B(), mode='a')
                     .band_pass_signals(low=5, high=40)
                     .update_variable('saved_batches', B(), mode='a'))
filter_pipeline = (eds >> filter_pipeline).run(batch_size=len(eds), n_epochs=1)

raw_batch, filtered_batch = filter_pipeline.get_variable('saved_batches')
#raw_batch.show_ecg('A00001', start=10, end=15)
#filtered_batch.show_ecg('A00001', start=10, end=15)

## Combining Pipelines
load_pipeline = ds.Pipeline().load(fmt="wfdb", components=["signal", "meta"])

filter_pipeline = ds.Pipeline().band_pass_signals(low=5, high=40)

augment_pipeline = (ds.Pipeline()
                      .random_resample_signals("normal", loc=300, scale=10)
                      .random_split_signals(length=3000, n_segments=5)
                      .unstack_signals())
full_pipeline = load_pipeline + filter_pipeline + augment_pipeline
with ds.Pipeline() as p:
    full_pipeline = (p.init_variable('saved_batches', init_on_each_run=list) +
                 full_pipeline +
                 p.update_variable('saved_batches', B(), mode='a'))
batch = (eds >> full_pipeline).run(batch_size=len(eds), n_epochs=1).get_variable('saved_batches')[0]
batch.show_ecg(0)
batch.show_ecg(5)
batch.show_ecg(29)


