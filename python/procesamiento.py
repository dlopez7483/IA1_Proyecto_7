import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_movie_lines(file_path):
    """
    Carga las líneas desde el archivo movie_lines.txt.
    Devuelve un diccionario con los IDs de las líneas como claves y el texto como valores.
    """
    lines = {}
    with open(file_path, encoding='UTF-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 3:  # Verificar que tiene los campos necesarios
                line_id = parts[0]
                text = parts[2]
                lines[line_id] = text
    return lines

def load_movie_conversations(file_path):
    """
    Carga las conversaciones desde el archivo movie_conversations.txt.
    Devuelve una lista de listas, donde cada sublista contiene los IDs de las líneas en una conversación.
    """
    conversations = []
    with open(file_path, encoding='UTF-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 2:  # Verificar que tiene los campos necesarios
                line_ids = parts[1][1:-1].replace("'", "").split(", ")  # Convertir a lista
                conversations.append(line_ids)
    return conversations

def preprocess_data(lines, conversations):
    """
    Procesa los datos de líneas y conversaciones para generar pares de preguntas y respuestas.
    También prepara los datos para el entrenamiento (tokenización y padding).
    """
    questions = []
    answers = []

    # Extraer pares de preguntas y respuestas
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            question_id = conversation[i]
            answer_id = conversation[i + 1]
            if question_id in lines and answer_id in lines:
                questions.append(lines[question_id])
                answers.append(lines[answer_id])

    # Limpiar el texto
    def clean_text(text):
        return text.lower().strip()

    questions_clean = [clean_text(q) for q in questions]
    answers_clean = [clean_text(a) for a in answers]

    # Tokenización
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions_clean + answers_clean)

    # Convertir texto a secuencias numéricas
    questions_seq = tokenizer.texts_to_sequences(questions_clean)
    answers_seq = tokenizer.texts_to_sequences(answers_clean)

    # Padding para que todas las secuencias tengan la misma longitud
    max_len_input = max(len(seq) for seq in questions_seq)
    max_len_output = max(len(seq) for seq in answers_seq)

    questions_seq = pad_sequences(questions_seq, maxlen=max_len_input, padding='post')
    answers_seq = pad_sequences(answers_seq, maxlen=max_len_output, padding='post')

    # Crear el vocabulario
    vocab_size = len(tokenizer.word_index) + 1  # +1 para el token <PAD>

    return questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size



def retornar_datos():
    # Definir rutas de los archivos
    lines_path = "movie_lines.txt"
    conversations_path = "movie_conversations.txt"

    # Verificar que los archivos existan
    if not os.path.exists(lines_path) or not os.path.exists(conversations_path):
        print("Error: Verifica las rutas de los archivos de entrada.")
        exit(1)

    # Cargar datos
    print("Cargando datos...")
    lines = load_movie_lines(lines_path)
    conversations = load_movie_conversations(conversations_path)

    # Procesar datos
    print("Procesando datos...")
    questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size = preprocess_data(lines, conversations)

    # Mostrar estadísticas de los datos procesados
    print(f"Número de pares de preguntas y respuestas: {len(questions_seq)}")
    print(f"Longitud máxima de entrada: {max_len_input}")
    print(f"Longitud máxima de salida: {max_len_output}")
    print(f"Tamaño del vocabulario: {vocab_size}")
    return questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size



############## anterior 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Flatten
from keras.models import Model
#import tensorflowjs as tfjs
import json
import string
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from googletrans import Translator

#Descarga los signos de puntuacion
import nltk
nltk.download('stopwords')

translator = Translator()

# Cargar los datos JSON
#with open('intents.json') as content:
#    data = json.load(content)

with open('intents.json', 'r', encoding='utf-8') as content:
    data = json.load(content)

# Cargar stopwords para inglés y español
stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))

tags = []
patterns = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

data = pd.DataFrame({"patterns": patterns, "tags": tags})
#data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
#data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

data['patterns'] = data['patterns'].apply(lambda wrd: [
    ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation
])
data['patterns'] = data['patterns'].apply(
    lambda wrd: ''.join(wrd)
)
data['patterns'] = data['patterns'].apply(
    lambda wrd: ' '.join([word for word in wrd.split() if word not in stop_words])
)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

x_train = pad_sequences(train)
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]

vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Corregir Input layer
i = Input(batch_shape=(None, input_shape))  # Modificado aquí
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(i, x)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200)

print(model.summary())

# Guardar el modelo en formato TensorFlow.js
#tfjs.converters.save_keras_model(model, './modelo_python')

# Guardar el tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(json.loads(tokenizer.to_json()), f)


def detectar_idioma(texto):
    # Detectar el idioma de la entrada
    detected_lang = translator.detect(texto).lang
    return detected_lang

def traducir_respuesta(respuesta, idioma_destino):
    # Traducir la respuesta al idioma detectado
    return translator.translate(respuesta, dest=idioma_destino).text


# Predicción interactiva
#while True:
#    texts_p = []
#    prediction_input = input('You: ')

#    prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
#    prediction_input = ''.join(prediction_input)
#    texts_p.append(prediction_input)

#    prediction_input = tokenizer.texts_to_sequences(texts_p)
#    prediction_input = np.array(prediction_input).reshape(-1)
#    prediction_input = pad_sequences([prediction_input], input_shape)

#    output = model.predict(prediction_input)
#    output = output.argmax()

#    response_tag = le.inverse_transform([output])[0]
#    print("Chatbot: ", random.choice(responses[response_tag]))


#este funciona bien
#while True:
#    texts_p = []
#    prediction_input = input('You: ')

#    prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
#    prediction_input = ''.join(prediction_input)

    # Remover stopwords
#    prediction_input = ' '.join([word for word in prediction_input.split() if word not in stop_words])

#    texts_p.append(prediction_input)

#    prediction_input = tokenizer.texts_to_sequences(texts_p)
#    prediction_input = pad_sequences(prediction_input, input_shape)

#    output = model.predict(prediction_input)
#    output = output.argmax()

#    response_tag = le.inverse_transform([output])[0]
#    print("Chatbot:", random.choice(responses[response_tag]))


while True:
    texts_p = []
    prediction_input = input('You: ')

    # Detectar el idioma de la entrada
    detected_lang = translator.detect(prediction_input).lang
    #print(f"Detected language: {detected_lang}")  # Imprimir idioma detectado para depuración
    
    # Si el idioma no es inglés ni español, lo traducimos al inglés
    if detected_lang not in ['en', 'es']:
        prediction_input = translator.translate(prediction_input, dest='en').text

    # Preprocesar texto (minúsculas y eliminar puntuación)
    prediction_input = [letter.lower() for letter in prediction_input if letter not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Convertir el texto a secuencias
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = pad_sequences(prediction_input, input_shape)

    # Hacer la predicción
    output = model.predict(prediction_input)
    output = output.argmax()

    # Obtener la etiqueta de respuesta
    response_tag = le.inverse_transform([output])[0]
    
    # Generar la respuesta en inglés
    response = random.choice(responses[response_tag])
    #print(f"Chatbot (English): {response}")

    # Si la entrada original estaba en español, traducir la respuesta a español
    if detected_lang == 'es':
        response = translator.translate(response, dest='es').text

    # Mostrar la respuesta final en el idioma original de la entrada
    print(f"Chatbot (Translated): {response}")















##############################3
from ast import literal_eval
import json
import string
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Flatten
from keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

# Cargar los datos del archivo intents.json
with open('intents.json') as content:
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
