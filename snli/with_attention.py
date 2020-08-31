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

premise = pad_sequences(premise,premise_max_lenght)
test_premise = pad_sequences(test_premise,premise_max_lenght)

hypothesis = pad_sequences(hypothesis,hypothesis_max_lenght)
test_hypothesis = pad_sequences(test_hypothesis,hypothesis_max_lenght)

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
    ,y)).shuffle(10000).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices(({
    "premise":test_premise,
    "hypothesis":test_hypothesis}
    ,test_y)).shuffle(1000).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

def make_model():

    l1 = keras.regularizers.l1

    l2 = keras.regularizers.l2

    premise_input = keras.layers.Input(premise.shape[0],name="premise")

    hypothesis_input = keras.layers.Input(hypothesis.shape[0],name="hypothesis")

    encode_model = keras.Sequential([
                                     keras.layers.Embedding(vocabulary_count+1,128,mask_zero=True),
                                     keras.layers.GaussianDropout(0.25),
                                     keras.layers.BatchNormalization()
                                     ])
                                     

    class dot_attention(keras.layers.Layer):
        def __init__(self,units):
            super(dot_attention,self).__init__()
            self.w = keras.layers.Dense(units,"swish")

        def call(self,quary,value):
            query_with_time_axis = tf.expand_dims(quary, 1)
            score = tf.matmul(query_with_time_axis,self.w(value))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * value
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights

    premise_vector = encode_model(premise_input)

    hypothesis_vector = encode_model(hypothesis_input)

    attention_layer = dot_attention(premise_vector.shape[1])

    # attention_vector_1 = keras.layers.GlobalAveragePooling1D()(attention_layer(([premise_vector,hypothesis_vector])))

    # attention_vector_2 = keras.layers.GlobalAveragePooling1D()(attention_layer(([hypothesis_vector,premise_vector])))

    attention_vector_1 = keras.layers.GlobalAveragePooling1D()(attention_layer(premise_vector,hypothesis_vector))

    attention_vector_2 = keras.layers.GlobalAveragePooling1D()(attention_layer(hypothesis_vector,premise_vector))

    attention_vector = keras.layers.Concatenate()([attention_vector_1,attention_vector_2])

    processing_model = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128,activation="swish",kernel_regularizer=l1(1e-4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64,"swish"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32,"swish"),
        keras.layers.Dropout(0.2),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16,"swish"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes_y,activation="softmax",kernel_regularizer=l2(1e-5),bias_initializer=base_bias)
    ])

    #processing_model = make_processer()

    out = processing_model(attention_vector)

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

epochs = 100

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