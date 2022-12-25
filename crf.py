from basic_func import *
from sklearn_crfsuite import CRF


def features(sentence, index, vec=[]):
    f = {
        'is_first_capital': int(sentence[index][0].isupper()),
        'prev_word': '' if index == 0 else sentence[index-1],
        'curr_word': sentence[index],
        'next_word': '' if index == len(sentence)-1 else sentence[index+1],
        'is_numeric': int(sentence[index].isdigit()),
        'prefix_1': sentence[index][0],
        'prefix_2': sentence[index][:2],
        'prefix_3': sentence[index][:3],
        'suffix_1':sentence[index][-1],
        'suffix_2':sentence[index][-2:],
        'suffix_3':sentence[index][-3:],
         }
    # optionally we can include word embedding dimensions as features (as shown below),
    # for a slight accuracy increase, but the time required to preprocess data hugely increases!
    """
    if sentence[index] not in vec:
        return f
    word_vec = vec[sentence[index]]
    for i in range(len(word_vec)):
        f['vector'+str(i)] = word_vec[i]
    """
    return f


def preprocess_crf (sentences):
    # import word2vec model if word embeddings are being used and pass it on function features
    # w2v = word2vec_words()
    x = []
    y = []
    for i in range(len(sentences)):
        temp_x = []
        temp_y = []
        for j in range(len(sentences[i])):
            temp_x.append(sentences[i][j][0])
            temp_y.append(sentences[i][j][1])
        fd_x = []
        for j in range(len(temp_x)):
            fd_x.append(features(temp_x, j))
        x.append(fd_x)
        y.append(temp_y)

    return x, y


def train_crf(x, y, con1=0.1, con2=0.1, max_ter=100):

    crf = CRF(
        c1=con1,
        c2=con2,
        max_iterations=max_ter,
        all_possible_transitions=True
    )

    crf.fit(x, y)

    return crf


def crf_model (train, con1, con2, max_ter):
    x_train, y_train = preprocess_crf(train)

    crf = train_crf(x_train, y_train, con1, con2, max_ter)

    return crf


if __name__ == "__main__":

    lang_choice = input("Choose language for CRF (1) English or (2) Greek: ")

    if lang_choice == '1':
        lang = "eng"
    else:
        lang = "gr"

    train, test = get_dataset(lang)

    # variables chosen in order to maximize accuracy
    c1, c2, max_ent = 0.1, 0.1, 150
    model = crf_model(train, c1, c2, max_ent)

    x_test, y_test = preprocess_crf(test)

    print("CRF Model Accuracy:", model.score(x_test, y_test))
