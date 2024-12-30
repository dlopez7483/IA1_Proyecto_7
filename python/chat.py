import json
import string
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator
import pandas as pd

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

# Crear el DataFrame para procesamiento adicional
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
translator = GoogleTranslator()


model = load_model('model.h5')

# Función para detectar el idioma
def detectar_idioma(mensaje):
    try:
        traducido = translator.translate(mensaje, target='en')
        if mensaje.lower() == traducido.lower():
            return 'en'
        else:
            return 'es'
    except:
        return 'es'  # Por defecto, asumir español


def mensaje (messaje):
 texts_p = []
 prediction_input = messaje

    # Detectar el idioma del usuario
 idioma_detectado = detectar_idioma(prediction_input)

    # Preprocesar el texto del usuario
 prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
 prediction_input = ''.join(prediction_input)
 texts_p.append(prediction_input)

 prediction_input = tokenizer.texts_to_sequences(texts_p)
 prediction_input = pad_sequences(prediction_input, maxlen=input_shape)

    # Predecir y obtener el intent
 output = model.predict(prediction_input)
 output = output.argmax()

 response_tag = le.inverse_transform([output])[0]

    # Seleccionar la respuesta en el idioma correcto
 if idioma_detectado in responses[response_tag]:
        response = random.choice(responses[response_tag][idioma_detectado])
 else:
        response = "Lo siento, no tengo una respuesta en tu idioma."

 return response

