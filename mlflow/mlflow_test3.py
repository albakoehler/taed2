from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from pathlib import Path
import pickle
import yaml
import mlflow

input_folder_path = Path("../src/data/processed")
output_folder_path = Path("models")

# Read training dataset
X_train = pd.read_csv(input_folder_path / "X_train.csv")
y_train = pd.read_csv(input_folder_path / "y_train.csv")
X_test = pd.read_csv(input_folder_path / "X_test.csv")
y_test = pd.read_csv(input_folder_path / "y_test.csv")

X_train.text=X_train.text.astype(str)
X_test.text=X_test.text.astype(str)

# Tokenizer      ========================================== #

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.text)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

# sequence models should be fed a sequence of numbers to it with no variance in shapes
# we'll use pad_sequence to make all sequences of a one constant length MAX_SEQUENCE_LENGTH
MAX_SEQUENCE_LENGTH = 30
x_train = pad_sequences(tokenizer.texts_to_sequences(X_train.text), maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(X_test.text), maxlen = MAX_SEQUENCE_LENGTH)

# label encoding
encoder = LabelEncoder()
encoder.fit(y_train.sentiment.to_list())
y_train = encoder.transform(y_train.sentiment.to_list())
y_test = encoder.transform(y_test.sentiment.to_list())
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# MODEL TRAINING ========================================== #

mlflow.tensorflow.autolog()

mlflow.set_experiment('Batch size value 512')

with mlflow.start_run(experiment_id=3):
    # parameters
    GLOVE_EMB = '../src/glove/glove.6B.300d.txt'
    EMBEDDING_DIM = 300
    LR = 0.001
    BATCH_SIZE = 512
    EPOCHS = 10
    EPOCHS = 12

    # Word embedding --- Embedding layer
    embeddings_index = {}
    f = open(GLOVE_EMB)
    for line in f:
        values = line.split()
        word = value = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                              EMBEDDING_DIM,
                                              weights=[embedding_matrix],
                                              input_length=MAX_SEQUENCE_LENGTH,
                                              trainable=False)

    # Building the model --- Embedding layer, Conv1D layer, LSTM and fully connected layers

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_sequences = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.2)(embedding_sequences)                     # embeddings
    x = Conv1D(64, 5, activation='relu')(x)                            # Conv1d layer
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x) #LSTM
    x = Dense(512, activation='relu')(x)                               # Fully connected layers
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(sequence_input, outputs)

    # Optimization algorithm --- Adam
    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy'])
    ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                             min_lr = 0.01,
                                             monitor = 'val_loss',
                                             verbose = 1)

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])

    #model.export_saved_model(output_folder_path)
