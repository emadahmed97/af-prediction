import sys, os
sys.path.append('..')

from cardio import EcgDataset
from cardio.pipelines import hmm_preprocessing_pipeline, hmm_train_pipeline
import warnings
warnings.filterwarnings('ignore')

SIGNALS_PATH = "C:/training2017/ECG/"
SIGNALS_MASK = os.path.join(SIGNALS_PATH, "*.hea")

dataset = EcgDataset(path=SIGNALS_MASK, no_ext=True, sort=True)
print(SIGNALS_MASK)

pipeline = hmm_preprocessing_pipeline()
ppl_inits = (dataset >> pipeline).run()

pipeline = hmm_train_pipeline(ppl_inits)
ppl_train = (dataset >> pipeline).run()

ppl_train.save_model("HMM", path="model_dump.dll")

eds = EcgDataset(path="C:/training2017/A*.hea", no_ext=True, sort=True)
batch = (eds >> hmm_predict_pipeline("model_dump.dll", annot="hmm_annotation")).next_batch(batch_size=3)

batch.show_ecg("A00001", start=10, end=15, annot="hmm_annotation")