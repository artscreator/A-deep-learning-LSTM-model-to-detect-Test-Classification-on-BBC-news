#%%
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os
import re


#%%
# 1) Data Loading
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"

df = pd.read_csv(URL)


#%%
# 2) Data Inspection
df.describe()
df.head()
df.info()

print(df['text'][0])


#%%
# 3) Data Cleaning
# 3.1) Remove numbers
# 3.2) Remove HTML Tags
# 3.3) Remove punctuation
# 3.4) Change all to lowercase()

for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('^\w.*?', '', data)
    df['text'][index] = re.sub('[^a-zA-Z]', ' ', df['text'][index]).lower()


#%%
# 4) Features Selection
text = df['text']
category = df['category']


#%%
# 5) Data Preprocessing

# Tokenizer
num_words = 5000 # unique number of words in all the sentences
oov_token = '<OOV>' # out of vocabrary

# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler() # instantiate
tokenizer = Tokenizer(num_words = num_words, oov_token=oov_token) # instantiate

# to train the tokenizer --> mms.fit(tokenizer)
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to transform the text using tokenizer --> mms.transform

text = tokenizer.texts_to_sequences(text)

# Padding
padded_text = pad_sequences(text, maxlen=300, padding='post', truncating='post')

# One hot encoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::,None])


#%%
# 6. Train test split
# expand dimension before feeding to train_test_split

padded_text = np.expand_dims(padded_text, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(padded_text, category, test_size = 0.2, random_state=123)


#%% 
# 7. Model development
embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words, embedding_layer))
model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.summary()

plot_model(model, show_shapes=True)

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])


#%%
# 8. TensorBoard

LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

ts_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

#%%
# 9. Perform model training
hist = model.fit(X_train,y_train,validation_data=(X_test, y_test), batch_size=32, epochs=30, callbacks=[ts_callback, es_callback])


# %% Model Analysis

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

y_predicted = model.predict(X_test)


#%%

y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))

# %% Model Saving

# to save trained model
model.save('model.h5')

# to save one hot encoder model
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# to save tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)