# Import required libraries
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.callbacks import Callback

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
# from keras import backend

import re
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm
# import seaborn as sns

seed = 7
seed = np.random.seed(seed)

# O model will be exported to this file
# filename='C:/analisesentimentoLSTM/model/model_saved.h5'
filename = 'model/model_saved.h5'

epochs = 60
# pre-workout word embedding dimensionality
word_embedding_dim = 60

# number of samples to use for each gradient update
batch_size = 40

# Reflects the maximum amount of words we will keep in the vocabulary
max_fatures = 6000

# Embedding layer output dimension
embed_dim = 128

# we limit the maximum length of all sentences
max_sequence_length = 300

pre_trained_wv = False

bilstm = False

# Clearing data for network input
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    cleanr = re.compile('<.*?>')
    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

    return string.strip().lower()

# Data preparation for network entry
def prepare_data(data):
    data = data[['texto', 'sentimento']]

    data['texto'] = data['texto'].apply(lambda x: x.lower())
    data['texto'] = data['texto'].apply(lambda x: clean_str(x))
    data['texto'] = data['texto'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    stop_words = set(stopwords.words('portuguese'))
    text = []
    for row in data["texto"].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    
    X = pad_sequences(X, maxlen=max_sequence_length)
    #X = pad_sequences(X)

    word_index = tokenizer.word_index
    Y = pd.get_dummies(data['sentimento']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,
                                                        random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=0.20,
                                                      random_state=42)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, word_index, tokenizer

# Loading data
data = pd.read_excel('basernrlstm2.xlsx')

X_train, X_test, X_val, Y_train, Y_test, Y_val, word_index, tokenizer = prepare_data(data)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Loading wordembeddings
def load_pre_trained_wv(word_index, num_words, word_embedding_dim):
    embeddings_index = {}
    f = open(os.path.join('./word_embedding', 'glove.6B.{0}d.txt'.format(word_embedding_dim)), encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('%s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((num_words, word_embedding_dim))
    for word, i in word_index.items():
        if i >= max_fatures:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Defining model
def model():
    if pre_trained_wv is True:
        print("USE PRE TRAINED")
        num_words = min(max_fatures, len(word_index) + 1)
        weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')    
        embedding = Embedding(
            num_words, 
            word_embedding_dim,
            input_length=max_sequence_length, 
            name="embedding", 
            weights=[weights_embedding_matrix], 
            trainable=False)(model_input)
        if bilstm is True:
            lstm = Bidirectional(LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    else:
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')    

        embedding = Embedding(max_fatures, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)
        
        if bilstm is True:
            lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
    
    model_output = Dense(2, activation='softmax', name="softmax")(lstm)
    model = Model(inputs=model_input, outputs=model_output)
    return model

model = model()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            
if not os.path.exists('./{}'.format(filename) ):

    hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs,
        batch_size=batch_size, shuffle=True, verbose=1)

    model.save_weights(filename)
    
#================================
    # Plot
    plt.figure()
    plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
    plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
    plt.title('Criminals classifier')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(hist.history['f1'], lw=2.0, color='b', label='train')
    plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
    plt.title('Criminals classifier')
    plt.xlabel('Epochs')
    plt.ylabel('f1_score')
    plt.legend(loc='upper left')
    plt.show()
   
else:
    model.load_weights('./{}'.format(filename))

scores = model.evaluate(X_test, Y_test, verbose = 0, batch_size = batch_size)
print("Acurácia: %.2f%%" % (scores[1]*100))

# Class with f1_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        #self.val_recalls = []
        #self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        #_val_recall = recall_score(val_targ, val_predict, average='macro')
        #_val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
        return

# Class Call
metric = Metrics()

# Run the model
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs,
    batch_size=batch_size, shuffle=True, verbose=1, callbacks=[metric])

# Plot
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('Criminals classifier')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(metric.val_f1s, lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('Criminals classifier')
plt.xlabel('Epochs')
plt.ylabel('f1_score')
plt.legend(loc='upper left')
plt.show()

#===========================================#
# Curva ROC
# pred_dicts = list(model.predict_proba(X_test))
# print (pred_dicts)

# probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

tmp = list(model.predict(X_test))
y_pred = []
for i in tmp:
    if i[0] > i[1]:
        y_pred.append(0)
    else:
        y_pred.append(1)
y_test = []
for i in Y_test:
    if i[0] > i[1]:
        y_test.append(0)
    else:
        y_test.append(1)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print ('Matriz de confusão')
print (cm)


"""
while True:
    sentence = input("input> ")

    if sentence == "exit":
        break
    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text, maxlen=max_sequence_length,
                             dtype='int32', value=0)

    sentiment = model.predict(new_text,batch_size=1,verbose = 2)[0]

    if(np.argmax(sentiment) == 0):
        pred_proba = "%.2f%%" % (sentiment[0]*100)
        print("crime => ", pred_proba)
    elif (np.argmax(sentiment) == 1):
        pred_proba = "%.2f%%" % (sentiment[1]*100)
        print("naocrime => ", pred_proba)
"""
