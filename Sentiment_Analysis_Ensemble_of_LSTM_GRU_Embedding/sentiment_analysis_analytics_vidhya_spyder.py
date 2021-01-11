import numpy as np
import pandas as pd

import tensorflow as tf
print(tf.__version__)

#tf.enable_eager_execution() #Not req if it's tf version 2.0
import os
os.chdir("C:\\Users\\kunjeshparekh\\Desktop\\KP\\IMS\\py\\project\\Sentiment_Analysis_Analytics_Vidhya")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train.info()
test.info()

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,0], train.iloc[:,1], train_size=0.8 , random_state=100)
X_train=train["tweet"]
y_train=train["label"]
X_test=test["tweet"]

train_np=np.array(X_train)
train_sent=np.array(y_train)
test_np=np.array(X_test)
#test_sent=np.array(y_test)
print(train_np[10:11])
#print(train_sent[0:2])
#print(test_np[0:2])
#print(test_sent[0:2])

vocab_size = 100000
embedding_dim = 42
max_length = 1000
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tkr = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tkr.fit_on_texts(train_np)
word_index = tkr.word_index
sequences = tkr.texts_to_sequences(train_np)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tkr.texts_to_sequences(test_np)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

np.max(padded) #24172which is less than 100000 here so vocab_size=100000 is good judgement

#Only Word Embedding with NN
model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#   tf.keras.layers.Flatten(),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_conv.summary()

#'''
#for non Embedding + LSTM use below
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#    tf.keras.layers.GlobalAveragePooling1D(),
#    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#    tf.keras.layers.Dense(72, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#For Embedding+GRU
model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#    tf.keras.layers.Dense(72, activation='relu'),
#    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_gru.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_gru.summary()

'''
model_sl = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])

model_sl.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_sl.summary()
'''

num_epochs = 5
model.fit(padded, np.array(y_train), epochs=num_epochs, batch_size=128, validation_data=(padded, np.array((y_train))))
num_epochs = 5
model_gru.fit(padded, np.array(y_train), epochs=num_epochs, batch_size=128, validation_data=(padded, np.array((y_train))))
num_epochs = 12
model_conv.fit(padded, np.array(y_train), epochs=num_epochs, batch_size=128, validation_data=(padded, np.array((y_train))))

predictions = model.predict(testing_padded)
predictions_gru = model_gru.predict(testing_padded)
predictions_conv = model_conv.predict(testing_padded)

print(testing_padded.shape)
print(predictions.shape)
predictions
label=np.where((predictions+predictions_gru+predictions_conv)>=1.5,1,0)

df_ans=pd.concat([test["id"],pd.DataFrame(label)],axis=1)
df_ans.columns=(["id","label"])
df_ans

#from google.colab import files
df_ans.to_csv('sentiment.csv',index=False) 
#files.download('sentiment.csv')

