from basic_func import *


train, test = get_dataset("eng")
print(len(train), len(test))
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

occurencies = []
for i in words:
    occurencies.append(words[i])
occurencies.sort(reverse=True)

zipf = []
for i in range(1,len(occurencies)):
    zipf.append(occurencies[0]/i)

plt.plot(occurencies)
plt.title("Word Frequency")
plt.ylabel('Times Encountered')
plt.xlabel('Words (leftmost is most frequent)')
plt.show()

plt.plot(occurencies)
plt.plot(zipf)
plt.title("Word Frequency - compared to Zipf's Law")
plt.ylabel('Times Encountered')
plt.xlabel('Words (leftmost is most frequent)')
plt.show()

occurencies = occurencies[:100]
plt.plot(occurencies)
plt.title("Word Frequency of 100 most used words")
plt.ylabel('Times Encountered')
plt.xlabel('Words')
plt.show()

plt.plot(occurencies)
plt.plot(zipf[:100])
plt.title("Word Frequency of 100 most used words - compared to Zipf's Law")
plt.ylabel('Times Encountered')
plt.xlabel('Words')
plt.show()

pos = list(tags_freq.keys())
freqs = list(tags_freq.values())

fig = plt.figure()

plt.bar(pos, freqs)
plt.xlabel("Part's of Speech Tagged with")
plt.ylabel("Number of occurencies")
plt.title("Frequency of POS Tags")
plt.show()

maxo = 0
mino = 0
meano = 0
for i in range(len(sentence_lengths)):
    crr = sentence_lengths[i]
    if crr > maxo:
        maxo = crr
    if crr < mino:
        mino = crr
    meano += crr
meano = meano / len(sentence_lengths)

print("Minimum length of a sentence is:", mino)
print("Maximum length of a sentence is:", maxo)
print("Mean length of a sentence is:", meano)
print("\nTotal number of words:", int(meano * len(sentence_lengths)))
print("Total number of unique words:",len(words))
print("\nAvailable part of speech tags:", tags)
print(tags_freq)
