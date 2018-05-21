import os
import tensorflow
from cardio import EcgDataset
import pandas as pd

PATH_TO_DATA = "C:/training2017"
pds = EcgDataset(path=os.path.join(PATH_TO_DATA, "*.hea"), no_ext=True, sort=True)
pds.cv_split(0.8, shuffle=True)

from cardio.pipelines import dirichlet_train_pipeline
os.environ["CUDA_VISIBLE_DEVICES"]="0"

AF_SIGNALS_REF = os.path.join(PATH_TO_DATA, "REFERENCE.csv")
pipeline = dirichlet_train_pipeline(AF_SIGNALS_REF, batch_size=256, n_epochs=500)

trained = (pds.train >> pipeline).run()

model_path = "af_model_dump"
trained.save_model("dirichlet", path=model_path)

pipeline = dirichlet_predict_pipeline(model_path)

res = (pds.test >> pipeline).run()
prediction = res.get_variable("predictions_list")

pd.options.display.float_format = '{:.2f}'.format

pred_proba = pd.DataFrame([x["target_pred"]["A"] for x in prediction], 
                    index=pds.test.indices, columns=['AF prob'])

true_labels = (pd.read_csv(AF_SIGNALS_REF, index_col=0, names=['True label'])
               .replace(['N', 'O', '~'], 'NO'))
               
df = pd.merge(pred_proba, true_labels, how='left', left_index=True, right_index=True)
print('hello!')
print(df)