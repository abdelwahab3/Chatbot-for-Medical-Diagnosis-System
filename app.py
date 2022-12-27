import json
import pickle
import random
from asyncio import run

import nltk
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')
intents = json.loads(open('data.json', encoding="utf8").read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def predict_response(msg):
    valus = msg.split(",")
    input_data = tuple(valus)
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # standardize the input data
    scaler2 = pickle.load(open("scaler.pkl", 'rb'))
    std_data = scaler2.transform(input_data_reshaped)

    model = pickle.load(open("diabetes.pkl", 'rb'))
    # 6,148,72,35,0,33.6,0.627,50
    prediction = model.predict(std_data)

    if (prediction[0] == 0):
        return "you r not diabetic"
    else:
        return "you r diabetic"


def chatbot_response(msg):
    if "," not in msg:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    else:
        res = predict_response(msg)
    return res


app = Flask(__name__)
app.static_folder = 'static'

# @app.route("/")
# def home():
#     return render_template("index.html")
#
#
# def chat():
#     return render_template("chatbot.html")
#
#
# app.add_url_rule("/chatbot.html", "chatbot", chat)
# app.add_url_rule("/index.html", "index", home)


# @app.route("/chat", methods=['POST'] )
# def get_bot_response():
#   userText = request.form['msg']
# return chatbot_response(userText)

# if __name__ == "__main__":
#   app.run(port=5000)

# connect between chatbot & website using flask
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'


@app.route("/chat", methods=['POST'])
def get_bot_response():
    userText = request.form['msg']
    return chatbot_response(userText)


if __name__ == "__main__":

    if __name__ == '__main__':
        app.run(port=5000)
