from basic_func import *
import numpy as np


def viterbi(words, tags, sent, p, a, b):
    if not sent:
        return []
    trellis = np.zeros((len(tags), len(sent)), dtype=float)
    pointers = np.zeros((len(tags), len(sent)), dtype=float)
    for i in range(len(sent)):
        if sent[i] not in words:
            sent[i] = -1
        else:
            sent[i] = words.index(sent[i])

    for i in range(len(tags)):
        trellis[i, 0] = p[i]*b[i, sent[0]]

    for i in range(1, len(sent)):
        for j in range(len(tags)):
            temp = []
            for h in range(len(tags)):
                temp.append(trellis[h, i-1]*a[h, j]*b[j, sent[i]])
            k = np.argmax(temp)

            trellis[j, i] = trellis[k, i-1]*a[k, j]*b[j, sent[i]]
            pointers[j, i] = k

    best_path = []
    temp = []
    for h in range(len(tags)):
        temp.append(trellis[h, len(sent)-1])

    k = np.argmax(temp)
    for i in range(len(sent)-1, -1, -1):
        k = int(k)
        best_path = [tags[k]] + best_path
        k = pointers[k, i]

    return best_path


def validate_sent (words, tags, sent, p, a, b):
    wds = [0]*len(sent)
    for i in range(len(wds)):
        wds[i] = sent[i][0]
    res = viterbi(words, tags, wds, p, a, b)
    corr = 0
    for i in range(len(res)):
        if res[i] == sent[i][1]:
            corr += 1

    return corr, len(sent)


def evaluate(test, words, tags, p, a, b):
    corr = 0
    tots = 0
    for i in range(len(test)):
        c, t = validate_sent(words, tags, test[i], p, a, b)
        corr += c
        tots += t
    return corr/tots


if __name__ == '__main__':

    lang_choice = input("Choose language for HMM (1) English or (2) Greek: ")

    if lang_choice == '1':
        lang = "eng"
    else:
        lang = "gr"

    train, test = get_dataset(lang)

    sentences = []
    tags = []
    words = {}
    sentence_lengths = []
    tags_freq = {}
    for i in range(len(train)):
        current = []
        for j in range(len(train[i])):
            current_word = train[i][j][0]
            current.append(current_word)
            if current_word in words:
                words[current_word] = words[current_word] + 1
            else:
                words[current_word] = 1
            current_tag = train[i][j][1]

            if current_tag not in tags:
                tags.append(current_tag)
                tags_freq[current_tag] = 1
            else:
                tags_freq[current_tag] += 1

        sentences.append(current)
        sentence_lengths.append(len(current))

    starting_prob = np.array([0]*len(tags), np.float32)
    tot_num = 0
    for i in range(len(train)):
        if len(train[i]):
            starting_prob[tags.index(train[i][0][1])] += 1
            tot_num += 1

    a = np.zeros((len(tags), len(tags)), dtype=float)
    for i in range(len(train)):
        for j in range(1, len(train[i])):
            prev = tags.index(train[i][j-1][1])
            curr = tags.index(train[i][j][1])
            a[prev, curr] += 1

    # Laplace smoothing
    for i in range(len(a)):
        for j in range(len(a[i])):
            # a[i, j] += 1
            pass

    occs = a.sum(axis=1)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i, j] = a[i, j] / occs[i]

    b = np.zeros((len(tags), len(words)+1), dtype=float)
    words = list(words.keys())
    for i in range(len(train)):
        for j in range(len(train[i])):
            word_curr = words.index(train[i][j][0])
            tag_curr = tags.index(train[i][j][1])
            b[tag_curr, word_curr] += 1

    occs = b.sum(axis=1)
    for i in range(len(tags)):
        # b[i, len(words)] = 1 # same propability for all tags
        b[i, len(words)] = occs[i] # same propability for all tags


    # Laplace smoothing, gives off worst results
    for i in range(len(b)):
        for j in range(len(b[i])):
            # b[i, j] += 1
            pass

    occs = b.sum(axis=1)
    for i in range(len(b)):
        for j in range(len(b[i])):
            b[i, j] = b[i, j] / occs[i]

    print(evaluate(test, words, tags, starting_prob, a, b))
