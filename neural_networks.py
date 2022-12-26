import os
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Bidirectional,SimpleRNN
from crf_layer import CRF
from basic_func import *


def rnn_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights,
              labels, lang):

    num_tags = len(labels)

    if lang == "eng":
        outd = 300
    else:
        outd = 96

    model = Sequential()
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=outd,
                        input_length=100))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(TimeDistributed(Dense(num_tags+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model_training = model.fit(x_train, y_train, batch_size=128, epochs=25,
                               validation_data=(x_validation, y_validation))

    loss, accuracy = model.evaluate(x_test, y_test)
    print("RNN model's accuracy:", accuracy)
    show_res(model_training, 'RNN Model Loss (across epochs)')

    return model, model_training


def lstm_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights,
               labels, lang):

    num_tags = len(labels)

    if lang == "eng":
        outd = 300
    else:
        outd = 96

    model = Sequential()
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=outd, input_length=100, weights=[embedding_weights]))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(num_tags+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    model_training = model.fit(x_train, y_train, batch_size=128, epochs=10,
                               validation_data=(x_validation, y_validation))

    loss, accuracy = model.evaluate(x_test, y_test)
    print("LSTM model's accuracy:", accuracy)
    show_res(model_training,  'LSTM Model Accuracy (across epochs)')

    return model, model_training


def bi_lstm_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                  embedding_weights, labels, lang):

    num_tags = len(labels)

    if lang == "eng":
        outd = 300
    else:
        outd = 96

    model = Sequential()
    # change output dim depending on language word2vec used, for english use 300, and for greek 96
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=outd, weights=[embedding_weights],
                        input_length=100))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_tags+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    model_training = model.fit(x_train, y_train, batch_size=128, epochs=10,
                               validation_data=(x_validation, y_validation))

    """
    preds = model.predict(x_test)
    corr = 0
    tots = 0
    for i in range(len(preds)):
        pred_tags = np.argmax(preds[i], axis=1)
        correct_tags = np.argmax(y_test[i], axis=1)
        for j in range(len(pred_tags)):
            tots += 1
            if pred_tags[j] == correct_tags[j]:
                corr += 1

    print(corr/tots, corr, tots)
    """

    loss, accuracy = model.evaluate(x_test, y_test)

    print("Bidirectional LSTM model's accuracy:", accuracy)
    show_res(model_training, 'Bidirectional LSTM Model Accuracy (across epochs)')

    return model, model_training


def complex_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                  embedding_weights, labels, lang):

    num_tags = len(labels)

    if lang == "eng":
        outd = 300
    else:
        outd = 96

    crf = CRF(num_tags+1, sparse_target=True)

    model = Sequential()
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=outd, weights=[embedding_weights],
                        input_length=100))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_tags + 1, activation='softmax')))
    model.add(crf)
    model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy])
    model.summary()

    model_training = model.fit(x_train, y_train, batch_size=128, epochs=5,
                               validation_data=(x_validation, y_validation))

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Bidirectional LSTM with CRF layer model's accuracy:", accuracy)
    show_res_crf(model_training, 'Bidirectional LSTM Model Accuracy (across epochs)')

    return model, model_training


def handle_rnn(lang, x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights, labels):

    path = lang + "/models/rnn_model"

    if os.path.exists(path):
        model = keras.models.load_model(path)
        print("Loaded RNN model!")
    else:
        model, model_training = rnn_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                                          embedding_weights, labels, lang)
        model.save(path)

    return model


def handle_lstm(lang, x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights, labels):

    path = lang + "/models/lstm_model"

    if os.path.exists(path):
        model = keras.models.load_model(path)
        print("Loaded LSTM model!")
    else:
        model, model_training = lstm_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                                          embedding_weights, labels, lang)
        model.save(path)

    return model


def handle_bi_lstm(lang, x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights, labels):

    path = lang + "/models/bi_lstm_model"

    if os.path.exists(path):
        model = keras.models.load_model(path)
        print("Loaded Bi-directional LSTM model!")
    else:
        model, model_training = bi_lstm_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                                              embedding_weights, labels, lang)
        model.save(path)

    return model


def handle_complex(lang, x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer, embedding_weights, labels):

    path = lang + "/models/complex_model"

    model, model_training = complex_model(x_train, y_train, x_validation, y_validation, x_test, y_test, word_tokenizer,
                                          embedding_weights, labels, lang)

    return model
