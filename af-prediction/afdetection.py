import sys, os
sys.path.append('..')

from cardio import EcgDataset
from cardio.pipelines import hmm_preprocessing_pipeline, hmm_train_pipeline, hmm_predict_pipeline
import warnings
warnings.filterwarnings('ignore')

SIGNALS_PATH = "ECG/" #set path to the QT database
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

dtst = EcgDataset(path=SIGNALS_MASK, no_ext=True, sort=True)

pipeline = hmm_preprocessing_pipeline()
ppl_inits = (dtst >> pipeline).run()

pipeline = hmm_train_pipeline(ppl_inits)
ppl_train = (dtst >> pipeline).run()

ppl_train.save_model("HMM", path="model_dump.dll")

# ECG Segmentation
eds = EcgDataset(path='cardio/tests/data/training2017/A*.hea', no_ext=True, sort=True)
batch = (eds >> hmm_predict_pipeline("model_dump.dll", annot="hmm_annotation")).next_batch()

batch.show_ecg("A00001", start=10, end=15, annot="hmm_annotation")

print("Heart rate: {0} bpm".format(int(.5 + batch["A00001"].meta["hr"])))

