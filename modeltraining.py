import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

sys.path.append(os.path.join("..", "..", ".."))
import cardio.dataset as ds
from cardio import EcgDataset
from cardio.dataset import B, V, F
from cardio.models.dirichlet_model import DirichletModel, concatenate_ecg_batch
from cardio.models.metrics import f1_score, classification_report, confusion_matrix

sns.set("talk")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

SIGNALS_PATH = "C:\\training2017\\"
SIGNALS_MASK = SIGNALS_PATH + "*.hea"
LABELS_PATH = SIGNALS_PATH + "REFERENCE.csv"

eds = EcgDataset(path=SIGNALS_MASK, no_ext=True, sort=True)
eds.cv_split(0.8)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)

model_config = {
    "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
    "input_shape": F(lambda batch: batch.signal[0].shape[1:]),
    "class_names": F(lambda batch: batch.label_binarizer.classes_),
    "loss": None,
}


N_EPOCH = 200
BATCH_SIZE = 256

template_train_ppl = (
    ds.Pipeline()
      .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)
      .init_variable("loss_history", init_on_each_run=list)
      .load(components=["signal", "meta"], fmt="wfdb")
      .load(components="target", fmt="csv", src=LABELS_PATH)
      .drop_labels(["~"])
      .rename_labels({"N": "NO", "O": "NO"})
      .flip_signals()
      .random_resample_signals("normal", loc=300, scale=10)
      .random_split_signals(2048, {"A": 9, "NO": 3})
      .binarize_labels()
      .train_model("dirichlet", make_data=concatenate_ecg_batch,
                   fetches="loss", save_to=V("loss_history"), mode="a")
      .call(lambda _, v: print(v[-1]), v=V('loss_history'))
      .run(batch_size=BATCH_SIZE, shuffle=True, drop_last=True, n_epochs=1, lazy=True)
)

train_ppl = (eds.train >> template_train_ppl).run()

train_loss = [np.mean(l) for l in np.array_split(train_ppl.get_variable("loss_history"), N_EPOCH)]

fig = plt.figure(figsize=(15, 4))
plt.plot(train_loss)
plt.xlabel("Epochs")
plt.ylabel("Training loss")
#plt.show()

MODEL_PATH = "C:\\training2017\\dirichlet_model"
train_ppl.save_model("dirichlet", path=MODEL_PATH)

template_test_ppl = (
    ds.Pipeline()
      .import_model("dirichlet", train_ppl)
      .init_variable("predictions_list", init_on_each_run=list)
      .load(components=["signal", "meta"], fmt="wfdb")
      .load(components="target", fmt="csv", src=LABELS_PATH)
      .drop_labels(["~"])
      .rename_labels({"N": "NO", "O": "NO"})
      .flip_signals()
      .split_signals(2048, 2048)
      .binarize_labels()
      .predict_model("dirichlet", make_data=concatenate_ecg_batch,
                     fetches="predictions", save_to=V("predictions_list"), mode="e")
      .run(batch_size=BATCH_SIZE, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
)

test_ppl = (eds.test >> template_test_ppl).run()

predictions = test_ppl.get_variable("predictions_list")

f1_score(predictions)
print(classification_report(predictions))
print(confusion_matrix(predictions))

uncertainty = [d["uncertainty"] for d in predictions]

fig = plt.figure(figsize=(15, 4))
sns.distplot(uncertainty, hist=True, norm_hist=True, kde=False)
plt.xlabel("Model uncertainty")
plt.xlim(-0.05, 1.05)
#plt.show()

q = 90
thr = np.percentile(uncertainty, q)
certain_predictions = [d for d in predictions if d["uncertainty"] <= thr]

f1_score(certain_predictions)

print(classification_report(certain_predictions))
print(confusion_matrix(certain_predictions))


BATCH_SIZE = 100


gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)

model_config = {
    "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
    "build": False,
    "load": {"path": MODEL_PATH},
}

template_predict_ppl = (
    ds.Pipeline()
      .init_model("static", DirichletModel, name="dirichlet", config=model_config)
      .init_variable("predictions_list", init_on_each_run=list)
      .load(fmt="wfdb", components=["signal", "meta"])
      .flip_signals()
      .split_signals(2048, 2048)
      .predict_model("dirichlet", make_data=partial(concatenate_ecg_batch, return_targets=False),
                     fetches="predictions", save_to=V("predictions_list"), mode="e")
      .run(batch_size=BATCH_SIZE, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
)

signal_name = "A00001.hea"
signal_path = SIGNALS_PATH + signal_name
predict_eds = EcgDataset(path=signal_path, no_ext=True, sort=True)
predict_ppl = (predict_eds >> template_predict_ppl).run()

predict_ppl.get_variable("predictions_list")

template_full_predict_ppl = (
    ds.Pipeline()
      .init_model("static", DirichletModel, name="dirichlet", config=model_config)
      .init_variable("signals", init_on_each_run=list)
      .init_variable("predictions_list", init_on_each_run=list)
      .init_variable("parameters_list", init_on_each_run=list)
      .load(fmt="wfdb", components=["signal", "meta"])
      .update_variable("signals", value=B("signal"))
      .flip_signals()
      .split_signals(2048, 2048)
      .predict_model("dirichlet", make_data=partial(concatenate_ecg_batch, return_targets=False),
                     fetches=["predictions", "parameters"],
                     save_to=[V("predictions_list"), V("parameters_list")], mode="e")
      .run(batch_size=BATCH_SIZE, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
)

def predict_and_visualize(signal_path):
    predict_eds = EcgDataset(path=signal_path, no_ext=True, sort=True)
    
    full_predict_ppl = (predict_eds >> template_full_predict_ppl).run()
    signal = full_predict_ppl.get_variable("signals")[0][0][0][:2000].ravel()
    predictions = full_predict_ppl.get_variable("predictions_list")[0]
    parameters = full_predict_ppl.get_variable("parameters_list")[0]
    
    print(predictions)

    x = np.linspace(0.001, 0.999, 1000)
    y = np.zeros_like(x)
    for alpha in parameters:
        y += beta.pdf(x, *alpha)
    y /= len(parameters)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2.5, 1]}, figsize=(15, 4))

    ax1.plot(signal)

    ax2.plot(x, y)
    ax2.fill_between(x, y, alpha=0.3)
    ax2.set_ylim(ymin=0)

    #plt.show()

predict_and_visualize(SIGNALS_PATH + "A00150.hea")
predict_and_visualize(SIGNALS_PATH + "A01505.hea")