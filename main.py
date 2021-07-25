from tensorflow.keras.datasets import imdb

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


    if not index_range: print(' '.join(id_to_word[id] for id in x_train[index] ))
    else:
        for i in range(range_start, range_end):
            print(' '.join(id_to_word[id] for id in x_train[i]))


if __name__ == '__main__':
    explore_data(159)
    print()
    explore_data(0, 159, 161)