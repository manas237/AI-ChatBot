import nltk  # Natural Language Tool Kit
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tensorflow
from tensorflow import keras
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []  # list of words in all the sentences(docs_x)
    labels = []  # unique tags
    docs_x = []  # list containing list of words
    docs_y = []  # all tags

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # Converts a sentence into a list containing its words
            words.extend(wrds)  # single strings are appended to a list using .append(). list of strings are appended to another list using .extend() - 1Dimensional
            docs_x.append(wrds)  # List containing lists of words grouped by each sentence(pattern) - 2Dimensional
            docs_y.append(intent["tag"])  # List containing tags corresponding to the sentence(pattern) - 1Dimensional

        if intent["tag"] not in labels:
            labels.append(intent["tag"])  # Creating 1Dimensional list containing unique tags

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # The sentences could very likely contain '?', need to exclude them
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(len(training[0]), )))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(len(output[0]), activation="softmax"))

model.compile(optimizer="adam", loss="mean_squared_error")


try:
    model.load("model.h5")
except:
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save("model.h5")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(numpy.array([bag_of_words(inp, words)]))
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat()
