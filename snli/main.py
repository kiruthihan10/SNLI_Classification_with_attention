import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import pad_sequences

[train_data,test_data],info = tfds.load("snli",split=["train","test"],batch_size=-1,with_info=True)

y = train_data["label"]
test_y = test_data["label"]

premise = train_data["premise"]
test_premise = test_data["premise"]

hypothesis = train_data["hypothesis"]
test_hypothesis = test_data["hypothesis"]

y= np.array(y)
test_y = np.array(test_y)

def base_bias(shape,dtype=None):
  return np.log(1/3*np.ones(shape))

num_classes_y = 3
y = keras.utils.to_categorical(y,num_classes=num_classes_y)
test_y = keras.utils.to_categorical(test_y,num_classes=num_classes_y)

print("Label Shape is {}".format(y.shape))

def converter(arr):
    arr = np.array(arr)
    new_arr = []
    for i in range(arr.shape[0]):
        new_line = arr[i].decode(encoding="UTF-8",errors='ignore')
        new_arr.append(new_line)
    return np.array(new_arr)

premise = converter(premise)
hypothesis = converter(hypothesis)

test_premise = converter(test_premise)
test_hypothesis = converter(test_hypothesis)

del train_data,test_data

enc = keras.preprocessing.text.Tokenizer()

enc.fit_on_texts(list(premise)+list(hypothesis))

vocabulary_count = len(enc.word_index)

premise = np.array(enc.texts_to_sequences(premise))
test_premise = np.array(enc.texts_to_sequences(test_premise))

hypothesis = np.array(enc.texts_to_sequences(hypothesis))
test_hypothesis = np.array(enc.texts_to_sequences(test_hypothesis))

premise_max_lenght = max(len(x) for x in list(premise) ) 
hypothesis_max_lenght = max(len(x) for x in list(hypothesis) ) 
max_lenght = np.amax([premise_max_lenght,hypothesis_max_lenght])

premise = pad_sequences(premise,max_lenght)
test_premise = pad_sequences(test_premise,max_lenght)

hypothesis = pad_sequences(hypothesis,max_lenght)
test_hypothesis = pad_sequences(test_hypothesis,max_lenght)

print("Hypothesis Shape is {}".format(hypothesis.shape))

print("premise Shape is {}".format(premise.shape))


#probs = np.bincount(y)
#probs = probs/y.shape[0]
#probs = probs/(1-probs)
#def base_bias(shape,dtype=None):
  #return np.log(probs)

batch_size = 64

ds = tf.data.Dataset.from_tensor_slices(({
    "premise":premise,
    "hypothesis":hypothesis}
    ,y)).shuffle(1024).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices(({
    "premise":test_premise,
    "hypothesis":test_hypothesis}
    ,test_y)).shuffle(1024).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

def make_model():

    l1 = keras.regularizers.l1

    l2 = keras.regularizers.l2

    vector_dim = 128

    premise_input = keras.layers.Input(premise.shape[1],name="premise")

    hypothesis_input = keras.layers.Input(hypothesis.shape[1],name="hypothesis")

    encode_model = keras.Sequential([
                                     keras.layers.Embedding(vocabulary_count+1,64,mask_zero=True),
                                     keras.layers.GaussianDropout(0.25),
                                     keras.layers.Conv1D(64,3,activation="tanh",kernel_regularizer=l2(1e-5)),
                                     keras.layers.GaussianDropout(0.25),
                                     keras.layers.Conv1D(64,3,2,activation="tanh",kernel_regularizer=l2(1e-5)),
                                     keras.layers.GaussianDropout(0.25),
                                     keras.layers.LSTM(128,kernel_regularizer=l1(1e-5),recurrent_regularizer=l1(1e-4),recurrent_dropout=0.2,return_sequences=True),
                                     keras.layers.GaussianDropout(0.225),
                                     keras.layers.LSTM(32,kernel_regularizer=l1(1e-5),recurrent_regularizer=l1(1e-4),recurrent_dropout=0.2),
                                     keras.layers.GaussianDropout(0.215),
                                     keras.layers.Dense(64,activation="swish",kernel_regularizer=l2(1e-5)),
                                     keras.layers.GaussianDropout(0.1)
    ])
    # def make_encoder():
    #     inp_1 = keras.layers.Input(premise.shape[1])
    #     emb = keras.layers.Embedding(vocabulary_count+1,64,mask_zero=True)(inp_1)
    #     emb = keras.layers.Conv1D(64,3,activation="tanh")(emb)
    #     emb = keras.layers.Conv1D(64,3,2,activation="tanh")(emb)
    #     rnn = keras.layers.LSTM(vector_dim)(emb)
    #     rnn = keras.layers.GaussianDropout(0.225)(rnn)
    #     d1 = keras.layers.Dense(vector_dim,activation="swish")(rnn)
    #     d1 = keras.layers.GaussianDropout(0.215)(d1)
    #     d2 = keras.layers.Dense(vector_dim,activation="swish")(d1)
    #     d2 = keras.layers.GaussianDropout(0.215)(d2)
    #     return keras.Model(inp_1,d2)
    # encode_model = make_encoder()

    premise_vector = encode_model(premise_input)

    hypothesis_vector = encode_model(hypothesis_input)

    #vector_difference = tf.math.sqrt(tf.math.squared_difference(premise_vector,hypothesis_vector))
    #vector_difference = tf.reshape(vector_difference,(batch_size,1))

    vector_difference = tf.math.abs(tf.math.subtract(premise_vector,hypothesis_vector))
    #vector_difference = keras.activations.tanh(vector_difference)

    # def make_processer():
    #     inp_2 = keras.layers.Input(vector_dim)
    #     norm = keras.layers.BatchNormalization()(inp_2)
    #     norm = keras.layers.Dropout(0.2)(norm)
    #     d3 = keras.layers.Dense(128,"swish")(norm)
    #     d3 = keras.layers.Dropout(0.2)(d3)
    #     d4 = keras.layers.Dense(128,"swish")(d3)
    #     d4 = keras.layers.BatchNormalization()(d4)
    #     d4 = keras.layers.Dropout(0.2)(d4)
    #     add_3 = keras.activations.swish(tf.math.add(d4,norm))
    #     add_3 = keras.layers.BatchNormalization()(add_3)
    #     prob_1 = keras.layers.Dense(3,activation="softmax",bias_initializer=base_bias)(add_3)
    #     return keras.Model(inp_2,prob_1)

    processing_model = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1,"swish",kernel_regularizer=l1(1e-4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32,activation="tanh",kernel_regularizer=l2(1e-5)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64,"relu",kernel_regularizer=l1(1e-4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32,"swish",kernel_regularizer=l2(1e-5)),
        keras.layers.Dropout(0.2),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes_y,activation="softmax",kernel_regularizer=l2(1e-5),bias_initializer=base_bias)
    ])

    #processing_model = make_processer()

    out = processing_model(vector_difference)

    model = keras.Model([premise_input,hypothesis_input],out)
    return model

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = make_model()

print(model.summary())

lr = 1e-3

opt = keras.optimizers.Nadam(lr)

model.compile(optimizer=opt,loss=keras.losses.CategoricalCrossentropy(label_smoothing=1e-9),metrics=["acc","mse"])

#model.compile(optimizer=opt,loss="mse",metrics=["acc"])

epochs = 10

decay = keras.optimizers.schedules.PolynomialDecay(lr,epochs,power=0.5)

schedule = keras.callbacks.LearningRateScheduler(decay)

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",patience=20,restore_best_weights=True)

history = model.fit(ds,validation_data=test_ds,epochs=epochs,workers=6,use_multiprocessing=True,callbacks=[schedule,early_stop])

model.evaluate(test_ds)

print(model.predict([test_premise,test_hypothesis]))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = list(range(int(len(loss))))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

val_acc = history.history['val_acc']
acc = history.history['acc']
epochs_range = list(range(int(len(loss))))
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation accuracy')
plt.show()