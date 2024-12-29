from ast import literal_eval
import json
import string
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Flatten
from keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

# Cargar los datos del archivo intents.json
with open('intents.json', 'r', encoding='utf-8') as content:
    data = json.load(content)

# Preprocesar los datos
tags = []
patterns = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

data = pd.DataFrame({"patterns": patterns, "tags": tags})
data['patterns'] = data['patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

# Configurar el tokenizador
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

x_train = pad_sequences(train)
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Construir el modelo
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(i, x)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Entrenar y guardar el modelo
model.fit(x_train, y_train, epochs=200)
model.save('model.h5')

# Configurar el traductor
translator = Translator()

# Chatbot en acción
while True:
    texts_p = []
    prediction_input = input("You: ")

    # Detectar el idioma del usuario
    detected_lang = translator.detect(prediction_input).lang

    # Traducir al inglés para predecir
    if detected_lang != 'en':
        prediction_input_translated = translator.translate(
            prediction_input, dest='en').text
    else:
        prediction_input_translated = prediction_input

    # Preprocesar texto traducido
    prediction_input_translated = [letter.lower()
                                   for letter in prediction_input_translated if letter not in string.punctuation]
    prediction_input_translated = ''.join(prediction_input_translated)
    texts_p.append(prediction_input_translated)

    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = pad_sequences(prediction_input, input_shape)

    # Predecir y obtener la respuesta
    output = model.predict(prediction_input)
    output = output.argmax()

    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])

    # Traducir la respuesta al idioma detectado
    if detected_lang != 'en':
        response = translator.translate(response, dest=detected_lang).text

    print("Chatbot:", response)
