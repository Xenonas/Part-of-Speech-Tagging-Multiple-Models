from basic_func import *
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.models import Sequential



def preprocess_char(data, tk=0, tagtk=0):
    x_data, y_data = split_data(data)

    x_new = []
    y_new = []

    for i in range(len(x_data)):
        for j in range(len(x_data[i])):
            x_new.append(x_data[i][j])
            y_new.append(y_data[i][j])

    if tk == 0:
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tk.fit_on_texts(x_new)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        char_dict = {}
        for i, char in enumerate(alphabet):
            char_dict[char] = i

        tk.word_index = char_dict.copy()
        tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    x = tk.texts_to_sequences(x_new)
    x = pad_sequences(x, maxlen=64, padding='post')

    if tagtk==0:
        tagtk = Tokenizer()
        tagtk.fit_on_texts(y_new)

    y = tagtk.texts_to_sequences(y_new)
    y = pad_sequences(y, maxlen=64, padding="pre", truncating="post")
    y = to_categorical(y)

    return x, y, tk, tagtk


train = read_data(r"D:\Downloads\Universal Dependencies 2.10\ud-treebanks-v2.10\UD_English-EWT\en_ewt-ud-train.conllu")
test = read_data(r"D:\Downloads\Universal Dependencies 2.10\ud-treebanks-v2.10\UD_English-EWT\en_ewt-ud-test.conllu")

x_train, y_train, tker, tag_tker = preprocess_char(train)
x_test, y_test, tker, tag_tker = preprocess_char(test, tker, tag_tker)

input_size = 64
vocab_size = len(tker.word_index)
embedding_size = 95
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1]]

fully_connected_layers = [1024, 1024]
num_tags = 18
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

embedding_weights = []
embedding_weights.append(np.zeros(vocab_size))

for char, i in tker.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

print(x_train.shape, y_train.shape)
model = Sequential()
model.add(Embedding(input_dim=len(tker.word_index) + 1, output_dim=300,
                    input_length=64))
model.add(Conv1D(64, kernel_size=7))
model.add(Dense(num_tags, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

model_training = model.fit(x_train, y_train, batch_size=128, epochs=8)

loss, accuracy = model.evaluate(x_test, y_test)