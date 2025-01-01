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

class Chatbot:
    def __init__(self, intents_file, model_file):
        self.intents_file = intents_file
        self.model_file = model_file
        self.responses = {}
        self.tokenizer = None
        self.le = None
        self.input_shape = None
        self.translator = GoogleTranslator()
        self.model = None
        self._load_data()
        self._prepare_model()

    def _load_data(self):
        with open(self.intents_file, 'r', encoding='utf-8') as content:
            data = json.load(content)

        # Preprocesar los datos
        tags = []
        patterns = []
        for intent in data['intents']:
            self.responses[intent['tag']] = intent['responses']
            for line in intent['patterns']:
                patterns.append(line)
                tags.append(intent['tag'])

        # Crear el DataFrame para procesamiento adicional
        df = pd.DataFrame({"patterns": patterns, "tags": tags})
        df['patterns'] = df['patterns'].apply(
            lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
        df['patterns'] = df['patterns'].apply(lambda wrd: ''.join(wrd))

        # Configurar el tokenizador
        self.tokenizer = Tokenizer(num_words=2000)
        self.tokenizer.fit_on_texts(df['patterns'])

        train = self.tokenizer.texts_to_sequences(df['patterns'])
        self.x_train = pad_sequences(train)

        self.le = LabelEncoder()
        self.y_train = self.le.fit_transform(df['tags'])

        self.input_shape = self.x_train.shape[1]

    def _prepare_model(self):
        self.model = load_model(self.model_file)

    def detectar_idioma(self, mensaje):
        try:
            traducido = self.translator.translate(mensaje, target='en')
            if mensaje.lower() == traducido.lower():
                return 'en'
            else:
                return 'es'
        except:
            return 'es'  # Por defecto, asumir español

    def responder(self, mensaje):
            texts_p = []
            prediction_input = mensaje

            # Detectar el idioma del usuario
            idioma_detectado = self.detectar_idioma(prediction_input)
            print(f"[DEBUG] Idioma detectado: {idioma_detectado}")


            idioma_detectado = self.detectar_idioma(prediction_input)
            print(f"[DEBUG] Idioma detectado: {idioma_detectado}")

            prediction_input = self.tokenizer.texts_to_sequences(texts_p)
            prediction_input = pad_sequences(prediction_input, maxlen=self.input_shape)

            # Predecir y obtener el intent
            output = self.model.predict(prediction_input)
            output = output.argmax()

            response_tag = self.le.inverse_transform([output])[0]

            # Seleccionar la respuesta en el idioma correcto
            if idioma_detectado in self.responses.get(response_tag, {}):
                response = random.choice(self.responses[response_tag][idioma_detectado])
            else:
                response = "Lo siento, no tengo una respuesta en tu idioma."

            return response
