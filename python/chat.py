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
import re # Importar la librería de expresiones regulares


# Variables globales
responses = {}
tokenizer = None
le = None
input_shape = None
translator = GoogleTranslator()
model = None

# Cargar datos
intents_file = "intents4.json"  # Cambia esta ruta según corresponda
model_file = "model.h5"       # Cambia esta ruta según corresponda

with open(intents_file, 'r', encoding='utf-8') as content:
    data = json.load(content)

# Preprocesar los datos
print("[INFO] Preprocesando datos...")
tags = []
patterns = []
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

# Crear el DataFrame para procesamiento adicional
df = pd.DataFrame({"patterns": patterns, "tags": tags})
df['patterns'] = df['patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
df['patterns'] = df['patterns'].apply(lambda wrd: ''.join(wrd))

# Configurar el tokenizador
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df['patterns'])

train = tokenizer.texts_to_sequences(df['patterns'])
x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

input_shape = x_train.shape[1]

# Cargar el modelo
model = load_model(model_file)

def detectar_idioma(mensaje):
    try:
        traducido = translator.translate(mensaje, target='en')
        if mensaje.lower() == traducido.lower():
            return 'en'
        else:
            return 'es'
    except:
        return 'es'  # Por defecto, asumir español


def detectar_lenguajes_js_python(mensaje):
    # Expresión regular para detectar lenguajes de programación
    pattern_python = r'python'
    pattern_js = r'javascript|js'
    if re.search(pattern_python, mensaje, re.IGNORECASE):
     return 'python'
    elif re.search(pattern_js, mensaje, re.IGNORECASE):
     return 'js'




def responder(mensaje):
    if not mensaje.strip():
        return "No entendí el mensaje, ¿puedes intentarlo de nuevo?"

    # Detectar el idioma del usuario
    idioma_detectado = detectar_idioma(mensaje)
    print(f"[DEBUG] Idioma detectado: {idioma_detectado}")
    print (f"[DEBUG] Lenguaje detectado: {detectar_lenguajes_js_python(mensaje)}")

    # Preprocesar el mensaje
    mensaje_procesado = ''.join([ltrs.lower() for ltrs in mensaje if ltrs not in string.punctuation])
    texts_p = [mensaje_procesado]
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)

    # Validar el contenido de la entrada
    if prediction_input.shape[0] == 0:
        return "No entendí el mensaje, intenta escribir algo más claro."

    try:
        # Predecir y obtener el intent
        output = model.predict(prediction_input)
        output = output.argmax()
        response_tag = le.inverse_transform([output])[0]
    except Exception as e:
        print(f"[ERROR] Error al predecir respuesta: {e}")
        return "Lo siento, no puedo procesar tu mensaje en este momento."

    # Seleccionar la respuesta en el idioma correcto
    response_options = responses.get(response_tag, {})
    if idioma_detectado in response_options:
        """
        respuesta = random.choice(response_options[idioma_detectado])
        print(f"[DEBUG] Respuesta seleccionada: {respuesta.values()}")
        if detectar_lenguajes_js_python(mensaje) == 'python' and "python" in respuesta:
            response = respuesta["python"]
        elif detectar_lenguajes_js_python(mensaje) == 'js' and "js" in respuesta:
            response = respuesta["js"]
        else:
            response = respuesta
            
        """

        
        
        print(f"[DEBUG] Respuesta seleccionada: {response_options[idioma_detectado]}")
        print(f"[DEBUG] response: {response_options}")
        print(f"[DEBUG] tipo: " + str(type(response_options)))
        if detectar_lenguajes_js_python(mensaje) == 'python' and "python" in response_options[idioma_detectado]:
         print("Respuesta en Python")
         respuesta = response_options[idioma_detectado]["python"]
         response = random.choice(respuesta)
        elif detectar_lenguajes_js_python(mensaje) == 'js' and "js" in response_options[idioma_detectado]:
         print("Respuesta en JS")
         respuesta = response_options[idioma_detectado]["js"]
         response = random.choice(respuesta)
        else:
         print("Respuesta en otro lenguaje")    
         response = random.choice(response_options[idioma_detectado]) 
    else:
        
        response = "Lo siento, no tengo una respuesta en tu idioma."

    return response

