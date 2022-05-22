import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
import pickle

from tensorflow import keras

class TextPredictor:
    def __init__(self):

        #constants        
        self.inp_words = 3
        self.path = "\\Users\\Xenia\\Desktop\\prog\\python\\alterex\\ann_2022\\saved_model"
        self.tokenizer_path = "\\Users\\Xenia\\Desktop\\prog\\python\\alterex\\ann_2022\\tokenizers"
        self.model = ''
        self.load_model()
    
    def load_model(self, path=""):
        if path == "":
            path = self.path
        if os.listdir(path):
            self.model = keras.models.load_model(path)
        else:
            self.create_and_train_model()

    def buildPhrase(self, texts, str_len=2):
        res = texts
        tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))
        #maxWordsCount = 1000
        #tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
        #              lower=True, split=' ', char_level=False)
        #tokenizer.fit_on_texts([texts])
#
        #dist = list(tokenizer.word_counts.items())
        #tokenizer = Tokenizer()
        #tokenizer.fit_on_texts([texts])
        #Tokenizer.fit_on_texts([texts])

        #dist = list(Tokenizer.word_counts.items())
        data = tokenizer.texts_to_sequences([texts])[0]
        for i in range(str_len):
            # x = to_categorical(data[i: i + inp_words], num_classes=maxWordsCount)  # преобразуем в One-Hot-encoding
            # inp = x.reshape(1, inp_words, maxWordsCount)
            x = data[i: i + self.inp_words]
            inp = np.expand_dims(x, axis=0)

            pred = self.model.predict(inp)
            indx = pred.argmax(axis=1)[0]
            data.append(indx)
            print("index&", indx)
            res += " " + tokenizer.index_word[indx]  # дописываем строку

        return res

    def create_and_train_model(self):
        with open('text.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

        maxWordsCount = 1000
        tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                              lower=True, split=' ', char_level=False)
        tokenizer.fit_on_texts([texts])

        dist = list(tokenizer.word_counts.items())
        print(dist[:10])

        data = tokenizer.texts_to_sequences([texts])

        pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

        # res = to_categorical(data[0], num_classes=maxWordsCount)
        # print(res.shape)
        res = np.array( data[0] )

        inp_words = 3
        n = res.shape[0] - inp_words

        X = np.array([res[i:i + inp_words] for i in range(n)])
        Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

        model = Sequential()
        model.add(Embedding(maxWordsCount, 256, input_length = inp_words))
        model.add(SimpleRNN(128, activation='tanh'))
        model.add(Dense(maxWordsCount, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        history = model.fit(X, Y, batch_size=32, epochs=50)

        

        self.model = model
        model.save(self.path)

        #res = buildPhrase("я слышу шум")
        #print(res)

