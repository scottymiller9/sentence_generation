import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from nltk.tokenize import word_tokenize

# load ascii text and convert to lowercase
filename = "wizardOfOz.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
words = word_tokenize(raw_text)
# create mapping of unique words to integers
unique_words = sorted(list(set(words)))
word_to_int = dict((w, i) for i, w in enumerate(unique_words))
# summarize the loaded data
n_words = len(words)
n_unique_words = len(unique_words)
print("Total words:", n_words)
print("Total unique_words:", n_unique_words)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 15
input_words = []
output_words = []
for i in range(0, n_words - seq_length, 1):
    seq_in = words[i:i + seq_length]
    seq_out = words[i + seq_length]
    input_words.append([word_to_int[word] for word in seq_in])
    output_words.append(word_to_int[seq_out])
n_patterns = len(input_words)
print("Total Patterns:", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(input_words, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_unique_words)
# one hot encode the output variable
y = np_utils.to_categorical(output_words)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=50, batch_size=64, callbacks=callbacks_list)
