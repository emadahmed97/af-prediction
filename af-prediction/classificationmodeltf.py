import tensorflow as tf
import sys, os
sys.path.append(os.path.join(".."))
import cardio.dataset as ds

from cardio import EcgDataset
from cardio.dataset.models.tf import TFModel
from cardio.dataset import F
from cardio.dataset import Pipeline, V
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

AF_SIGNALS_PATH = 'cardio/tests/data/training2017/' #set path to the PhysioNet database
AF_SIGNALS_MASK = os.path.join(AF_SIGNALS_PATH, "*.hea")
AF_SIGNALS_REF = os.path.join(AF_SIGNALS_PATH, "REFERENCE.csv")

afds = EcgDataset(path=AF_SIGNALS_MASK, no_ext=True, sort=True)

afds.cv_split(0.8)

preprocess_pipeline = (ds.Pipeline()
                       .load(fmt="wfdb", components=["signal", "meta"])
                       .load(src=AF_SIGNALS_REF, fmt="csv", components="target")
                       .drop_labels(["~"])
                       .rename_labels({"N": "NO", "O": "NO"})

                       .drop_short_signals(4000)
                       .random_split_signals(3000, 3)
                       .apply_transform(func=np.transpose, src='signal', dst='signal', axes=[0, 2, 1]))

def tf_conv_block(input_layer, nb_filters, is_training):
        conv = tf.layers.conv1d(input_layer, nb_filters, 4)
        bnorm = tf.layers.batch_normalization(conv, training=is_training, momentum=0.9)
        relu = tf.nn.relu(bnorm)
        maxp = tf.layers.max_pooling1d(relu, 2, 2)
        return maxp

class TFConvModel(TFModel):
    def _build(self, config=None):
        with self:
            x = tf.placeholder("float", (None,) + self.config['input_shape'])
            self.store_to_attr("x", x)
            y = tf.placeholder("float", (None, self.config['n_classes']))
            self.store_to_attr("y", y)
            conv = x
            for nb_filters in self.config['filters']:
                conv = tf_conv_block(conv, nb_filters, self.is_training)
            flat = tf.reduce_max(conv, axis=1)
            logits = tf.layers.dense(flat, self.config['n_classes'])
            output = tf.nn.softmax(logits)
            self.store_to_attr("output", output)
            if self.config['loss'] == 'binary_crossentropy':
                loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            else:
                raise KeyError('loss {0} is not implemented'.format(self.config['loss']))

tf_model_config = {
    "input_shape": F(lambda batch: batch.signal[0].shape[1:]),
    "n_classes": 2,
    "loss": 'binary_crossentropy',
    "optimizer": "Adam",
    'filters': [4, 4, 8, 8, 16, 16, 32, 32]
}


def make_signals(batch):
    return np.array([segment for signal in batch.signal for segment in signal])

def make_targets(batch):
    n_reps = [signal.shape[0] for signal in batch.signal]
    return np.repeat(batch.target, n_reps, axis=0)

def tf_make_data(batch, **kwagrs):
    if any(elem is None for elem in batch.target):
        return {"feed_dict": {'x': make_signals(batch)}}
    else:
        return {"feed_dict": {'x': make_signals(batch), 'y': make_targets(batch)}}

with Pipeline() as p:
    tf_train_pipeline = (p.init_model("dynamic", TFConvModel, name="conv_model", config=tf_model_config)
                         +
                         p.init_variable("loss_history", init_on_each_run=list)
                         +
                         preprocess_pipeline
                         +
                         p.binarize_labels()
                         +
                         p.train_model('conv_model', make_data=tf_make_data, fetches="loss",
                                       save_to=V("loss_history"), mode="a"))

tf_model_trained = (afds.train >> tf_train_pipeline).run(batch_size=256, n_epochs=1, shuffle=True, drop_last=True)

plt.plot(tf_model_trained.get_variable("loss_history")[10:])
plt.xlabel("Iteration")
plt.ylabel("Training loss")
# plt.show()


tf_predict_pipeline = (ds.Pipeline()
                       .import_model("conv_model", tf_model_trained)
                       .init_variable("predictions_list", init_on_each_run=list)
                       .load(fmt="wfdb", components=["signal", "meta"])
                       .random_split_signals(3000, 1)
                       .apply_transform(func=np.transpose, src='signal', dst='signal', axes=[0, 2, 1])
                       .predict_model('conv_model', make_data=tf_make_data,
                                      fetches='output', save_to=V("predictions_list"), mode="e"))
tf_res = (afds.test >> tf_predict_pipeline).run(batch_size=len(afds.test.indices), n_epochs=1, shuffle=False, drop_last=False)

pd.options.display.float_format = '{:.2f}'.format

pred_proba = pd.DataFrame([x[0] for x in tf_res.get_variable("predictions_list")],
                    index=afds.test.indices, columns=['AF prob'])

true_labels = (pd.read_csv(AF_SIGNALS_REF, index_col=0, names=['True label'])
               .replace(['N', 'O', '~'], 'NO'))

df = pd.merge(pred_proba, true_labels, how='left', left_index=True, right_index=True)
df.head(10)

print(classification_report((df['True label'] == 'A').astype(int),
                            (df['AF prob'].astype(float) >= 0.5).astype(int),
                            target_names = ['NO', 'A']))