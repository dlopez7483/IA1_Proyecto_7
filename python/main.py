from procesamiento import preprocess_data, retornar_datos
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.utils import to_categorical

# Rutas de los archivos
movie_lines_path = "movie_lines.txt"
movie_conversations_path = "movie_conversations.txt"

# Procesar los datos
questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size = retornar_datos()

# Imprimir una muestra de las secuencias
print("Muestra de las secuencias de preguntas:")
print(questions_seq[:5])  # Primeras 5 secuencias de preguntas

print("Muestra de las secuencias de respuestas:")
print(answers_seq[:5])  # Primeras 5 secuencias de respuestas

print(f"Tamaño del vocabulario: {vocab_size}")

# Crear las secuencias de entrada para el decodificador (sin el último token)
decoder_input_seq = answers_seq[:, :-1]  # Las respuestas sin el último token

# Crear las secuencias de salida para el decodificador (sin el primer token)
decoder_output_seq = answers_seq[:, 1:]  # Las respuestas sin el primer token

# Verificar las dimensiones de las secuencias
print(f"Dimensiones de decoder_input_seq: {decoder_input_seq.shape}")
print(f"Dimensiones de decoder_output_seq: {decoder_output_seq.shape}")

# Convertir las secuencias de salida en one-hot encoding
decoder_output_seq_one_hot = np.zeros((len(answers_seq), max_len_output, vocab_size), dtype='float32')
for i, seq in enumerate(decoder_output_seq):
    for t, word in enumerate(seq):
        decoder_output_seq_one_hot[i, t, word] = 1.0

# Verificar las dimensiones de las secuencias en one-hot encoding
print(f"Dimensiones de decoder_output_seq_one_hot: {decoder_output_seq_one_hot.shape}")

# Convertir las respuestas a one-hot encoding
decoder_target_seq = to_categorical(answers_seq, num_classes=vocab_size)

# Definir el modelo
encoder_input_seq = Input(shape=(max_len_input,))
decoder_input_seq = Input(shape=(max_len_output,))

# Capa de embedding para el codificador
embedding = Embedding(input_dim=vocab_size, output_dim=256)(encoder_input_seq)

# LSTM encoder
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(embedding)

# Decodificador
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=256)(decoder_input_seq)
decoder_lstm = LSTM(256, return_sequences=True)(decoder_embedding, initial_state=[state_h, state_c])

# Capa densa para predecir la siguiente palabra
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_output = decoder_dense(decoder_lstm)

# Crear el modelo
model = Model([encoder_input_seq, decoder_input_seq], decoder_output)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 100  
batch_size = 32 

print("Entrenando el modelo...")


model.fit(
    [questions_seq, decoder_input_seq], 
    decoder_target_seq,  
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2  
)


model.save('chatbot_model.h5')
