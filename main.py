# Author: James Loy
# Studied & Modified by: Lee Taylor
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf


def explore_data(index=159, range_start=None, range_end=None):
    index_range = False
    if range_start != None and range_end != None:
        if range_end <= range_start:
            raise TypeError("Index range end number must be greater than starting number.")
        index_range = True
    training_set, testing_set = imdb.load_data(index_from=3)
    x_train, y_train = training_set
    x_test, y_test = testing_set

    print(x_train[0])

    word_to_id = imdb.get_word_index()
    word_to_id = {key:(value+3) for key,value in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    id_to_word = {value:key for key,value in word_to_id.items()}

    if not index_range:
        print(' '.join(id_to_word[id] for id in x_train[index] ))
        print(f"Sentiment Output: {y_train[index]}")
    else:
        for i in range(range_start, range_end):
            print(' '.join(id_to_word[id] for id in x_train[i]))
            print(f"Sentiment Output: {y_train[i]}")

def load_data():
    """ Loading data and padding """
    training_set, testing_set = imdb.load_data(num_words = 10000)
    x_train, y_train = training_set
    x_test, y_test = testing_set
    x_train_padded = sequence.pad_sequences(x_train, maxlen = 100)
    x_test_padded = sequence.pad_sequences(x_test, maxlen = 100)
    return x_train_padded, y_train, x_test_padded, y_test

def build_model(opt='SGD'):
    """ Build model """
    model = Sequential()
    model.add(Embedding(input_dim = 10000, output_dim = 128))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation='sigmoid'))
    # if verbose: model.summary()
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_test_model(model, data, model_fd=None, verbose=True):
    """ Score model. Params xv & yv = validation set """
    if not len(data) == 4:
        raise TypeError('Pass (x_train_p, y_train, x_test_p, y_test) in a tuple.')
    x, y, xv, yv = data
    scores = model.fit(x, y, batch_size=128, epochs=10,
                       validation_data=(xv, yv))
    if verbose: print(f"Training Accuracy: {scores.history['accuracy'].pop()}%")
    return scores

if __name__ == '__main__':
    """ Exploring imdb data """
    # explore_data(6)
    # print()
    # explore_data(0, 159, 161)
    """ Train and score model """
    data_tuple = load_data()
    # Stochastic gradient descent model
    sgd_model = build_model()
    sgd_scores = train_test_model(sgd_model, data_tuple)
    # RMSprop optimizer model
    rmsp_model = build_model('RMSprop')
    rmsp_scores = train_test_model(rmsp_model, data_tuple)
    # Adam optimizer model
    adam_model = build_model('adam')
    adam_scores = train_test_model(adam_model, data_tuple)
    pass