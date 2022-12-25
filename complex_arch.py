from basic_func import *
from neural_networks import *
from crf import preprocess_crf, train_crf


def get_features_crf(vector):
    ft = {}
    for i in range(len(vector)):
        ft['v'+str(i)] = vector[i]
    return ft


def prepare_complex_data(x, y, lstm_trained_model, dataset):
    y_pred = lstm_trained_model.predict(x)
    x_crf_train = []
    x_null, y_crf_train = preprocess_crf(dataset)

    for i in range(len(y_pred)):
        new_sent = []
        start_point = len(y_pred[i]) - len(y_crf_train[i])
        for j in range(start_point, len(y_pred[i])):
            new_sent.append(get_features_crf(y_pred[i][j]))
        x_crf_train.append(new_sent)

    return x_crf_train, y_crf_train


def train_crf_complex(x, y, lstm_trained_model, dataset):

    x_crf_train, y_crf_train = prepare_complex_data(x, y, lstm_trained_model, dataset)
    print(len(x_crf_train), len(x_crf_train[0]), len(x_crf_train[0][0]))
    crf = train_crf(x_crf_train, y_crf_train)

    return crf


lang = "eng"
train, test = get_dataset(lang)

labels = create_reference_list(train)

print("Started preprocessing...")
input_tr, out_tr, word_tokenizer, tag_tokenizer, embedding_weights = preprocess_data(train)
input_test, out_test, word_tokenizer, tag_tokenizer, embedding_weights \
        = preprocess_data(test, word_tokenizer, tag_tokenizer, embedding_weights)
input_tr, out_tr, input_val, out_val = create_val(input_tr, out_tr, 0.2)
print("Finished preprocessing!")

model = handle_bi_lstm(lang, input_tr, out_tr, input_val, out_val, input_test, out_test,
                       word_tokenizer, embedding_weights, labels)

crf = train_crf_complex(input_tr, out_tr, model, train)

x_test, y_test = prepare_complex_data(input_test, out_test, model, test)

print(len(x_test), len(y_test), len(x_test[0]), len(y_test[0]))

print("CRF Model Accuracy:", crf.score(x_test, y_test))
