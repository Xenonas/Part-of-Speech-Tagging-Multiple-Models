from neural_networks import *
from basic_func import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    while 1:
        lang_choice = input("Choose language (1) English or (2) Greek:")
        model_choice = input("Please choose (1) RNN POST model (2) LSTM POST model (3) Bi-Directional LSTM POST model "
                             "(4) Bi-Directional LSTM with CRF layer:")
        if lang_choice == '2':
            lang = "gr"
        else:
            lang = "eng"

        train, test = get_dataset(lang)
        labels = create_reference_list(train)

        print("Started preprocessing...")
        input_tr, out_tr, word_tokenizer, tag_tokenizer, embedding_weights = preprocess_data(train, language=lang)
        input_test, out_test, word_tokenizer, tag_tokenizer, embedding_weights \
            = preprocess_data(test, word_tokenizer, tag_tokenizer, embedding_weights, language=lang)
        input_tr, out_tr, input_val, out_val = create_val(input_tr, out_tr, 0.2)
        print("Finished preprocessing!")

        try:
            model_choice = int(model_choice)
        except ValueError:
            continue

        if model_choice == 1:
            final_model = handle_rnn(lang, input_tr, out_tr, input_val, out_val, input_test, out_test,
                                     word_tokenizer, embedding_weights, labels)
            break

        elif model_choice == 2:
            final_model = handle_lstm(lang, input_tr, out_tr, input_val, out_val, input_test, out_test,
                                      word_tokenizer, embedding_weights, labels)
            break

        elif model_choice == 3:
            final_model = handle_bi_lstm(lang, input_tr, out_tr, input_val, out_val, input_test, out_test,
                                         word_tokenizer, embedding_weights, labels)
            break

        elif model_choice == 4:
            final_model = handle_complex(lang, input_tr, out_tr, input_val, out_val, input_test, out_test,
                                         word_tokenizer, embedding_weights, labels)
            break

    while 1:
        s = input("Write the sentence you want tagged:")
        make_prediction(final_model, word_tokenizer, tag_tokenizer, s, labels)
