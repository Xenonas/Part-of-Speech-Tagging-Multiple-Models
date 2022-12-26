from basic_func import *


def naive_method(train_set, ref):
    occurrences = {}
    for i in range(0, len(train_set)):
        for j in range(0, len(train_set[i])):
            if not train_set[i][j][0] in occurrences:
                occurrences[train_set[i][j][0]] = [0]*len(ref)

            occurrences[train_set[i][j][0]][ref.index(train_set[i][j][1])] += 1

    return occurrences


def two_gram(train_set, ref):
    occurrences = {}
    for i in range(1, len(train_set)):
        for j in range(1, len(train_set[i])):
            word = str(train_set[i][j-1][0] + '-' + train_set[i][j][0])
            if word not in occurrences:
                occurrences[word] = [0]*len(ref)
            occurrences[word][ref.index(train_set[i][j][1])] += 1
    return occurrences


def test_two_gram(occurrences, eval_data, ref):
    correct_guesses = 0
    total_guesses = 0
    adjusted_total_guesses = 0
    for i in range(1, len(eval_data)):
        for j in range(1, len(eval_data[i])):
            total_guesses += 1
            word = str(eval_data[i][j-1][0] + '-' + eval_data[i][j][0])
            if not (word in occurrences):
                continue
            elif occurrences[word].index(max(occurrences[word])) == ref.index(eval_data[i][j][1]):
                adjusted_total_guesses += 1
                correct_guesses += 1
            else:
                adjusted_total_guesses += 1
    print(correct_guesses / total_guesses, correct_guesses / adjusted_total_guesses)
    return correct_guesses, total_guesses, adjusted_total_guesses


def three_gram(train_set, ref):
    occurrences = {}
    for i in range(2, len(train_set)):
        for j in range(2, len(train_set[i])):
            word = str(train_set[i][j-2][0] + '-' + train_set[i][j-1][0] + '-' + train_set[i][j][0])
            if word not in occurrences:
                occurrences[word] = [0]*len(ref)
            occurrences[word][ref.index(train_set[i][j][1])] += 1
    return occurrences


def test_three_gram(occurrences, eval_data, ref):
    correct_guesses = 0
    total_guesses = 0
    adjusted_total_guesses = 0
    for i in range(2, len(eval_data)):
        for j in range(2, len(eval_data[i])):
            total_guesses += 1
            word = str(eval_data[i][j-2][0] + '-' + eval_data[i][j-1][0] + '-' + eval_data[i][j][0])
            if not (word in occurrences):
                continue
            elif occurrences[word].index(max(occurrences[word])) == ref.index(eval_data[i][j][1]):
                adjusted_total_guesses += 1
                correct_guesses += 1
            else:
                adjusted_total_guesses += 1
    print(correct_guesses / total_guesses, correct_guesses / adjusted_total_guesses)
    return correct_guesses, total_guesses, adjusted_total_guesses


def test_naive(occurrences, eval_data, ref):
    correct_guesses = 0
    total_guesses = 0
    adjusted_total_guesses = 0
    for i in range(0, len(eval_data)):
        for j in range(0, len(eval_data[i])):
            total_guesses += 1
            if not (eval_data[i][j][0] in occurrences):
                continue
            elif occurrences[eval_data[i][j][0]].index(max(occurrences[eval_data[i][j][0]])) \
                    == ref.index(eval_data[i][j][1]):
                adjusted_total_guesses += 1
                correct_guesses += 1
            else:
                print(ref[occurrences[eval_data[i][j][0]].index(max(occurrences[eval_data[i][j][0]]))], eval_data[i][j][1], eval_data[i][j][0], eval_data[i])
                adjusted_total_guesses += 1
    print(correct_guesses/total_guesses, correct_guesses/adjusted_total_guesses)
    return correct_guesses, total_guesses, adjusted_total_guesses


def test_combined_naive(occurrences_single, occurrences_two, eval_data, ref):
    correct_guesses = 0
    total_guesses = 0
    adjusted_total_guesses = 0
    for i in range(1, len(eval_data)):
        for j in range(1, len(eval_data[i])):
            total_guesses += 1
            word = str(eval_data[i][j-1][0] + '-' + eval_data[i][j][0])
            if not (word in occurrences_two):
                if not (eval_data[i][j][0] in occurrences_single):
                    continue
                elif occurrences_single[eval_data[i][j][0]].index(max(occurrences_single[eval_data[i][j][0]])) \
                        == ref.index(eval_data[i][j][1]):
                    adjusted_total_guesses += 1
                    correct_guesses += 1
                else:
                    adjusted_total_guesses += 1
            elif occurrences_two[word].index(max(occurrences_two[word])) == ref.index(eval_data[i][j][1]):
                adjusted_total_guesses += 1
                correct_guesses += 1
            else:
                adjusted_total_guesses += 1
    print(correct_guesses / total_guesses, correct_guesses / adjusted_total_guesses)
    return correct_guesses, total_guesses, adjusted_total_guesses


if __name__ == '__main__':

    train, test = get_dataset("gr")
    labels = create_reference_list(train)

    occ = naive_method(train, labels)
    occ2 = two_gram(train, labels)
    occ3 = three_gram(train, labels)

    print("Accuracy for unigram model:")
    test_naive(occ, test, labels)
    print("\nAccuracy for bigram model:")
    test_two_gram(occ2, test, labels)
    print("\nAccuracy for trigram model:")
    test_three_gram(occ3, test, labels)
    print("\nAccuracy for best-gram model:")
    test_combined_naive(occ, occ2, test, labels)
