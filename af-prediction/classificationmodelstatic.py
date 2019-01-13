import sys, os
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from cardio import EcgDataset
from cardio.pipelines import dirichlet_train_pipeline
from cardio.pipelines import dirichlet_predict_pipeline
import pandas as pd
from sklearn.metrics import classification_report

AF_SIGNALS_PATH = 'cardio/tests/data/training2017/' #set path to the PhysioNet database
AF_SIGNALS_MASK = os.path.join(AF_SIGNALS_PATH, "*.hea")
AF_SIGNALS_REF = os.path.join(AF_SIGNALS_PATH, "REFERENCE.csv")

afds = EcgDataset(path=AF_SIGNALS_MASK, no_ext=True, sort=True)

afds.cv_split(0.8)

pipeline = dirichlet_train_pipeline(AF_SIGNALS_REF, n_epochs=1)
train_ppl = (afds.train >> pipeline).run()

train_loss = train_ppl.get_variable("loss_history")

fig = plt.figure(figsize=(15, 4))
plt.plot(train_loss)
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.show()

train_ppl.save_model("dirichlet", path="af_model_dump")

pipeline = dirichlet_predict_pipeline(model_path="af_model_dump")
res = (afds.test >> pipeline).run()

pred = res.get_variable("predictions_list")

pd.options.display.float_format = '{:.2f}'.format

pred_proba = pd.DataFrame([x["target_pred"]["A"] for x in pred],
                    index=afds.test.indices, columns=['AF prob'])

true_labels = (pd.read_csv(AF_SIGNALS_REF, index_col=0, names=['True label'])
               .replace(['N', 'O', '~'], 'NO'))

df = pd.merge(pred_proba, true_labels, how='left', left_index=True, right_index=True)
df.head(10)


print(classification_report((df['True label'] == 'A').astype(int),
                            (df['AF prob'].astype(float) >= 0.5).astype(int)))