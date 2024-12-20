import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as readline from 'readline';

// Cargar el JSON del modelo y del tokenizer
const modelJson = JSON.parse(fs.readFileSync('modeloo.json')); // Cargar el modelo desde un archivo JSON
const tokenizerJson = JSON.parse(fs.readFileSync('tokenizer.json')); // Cargar el tokenizer

const wordIndex = tokenizerJson['config']['word_index']; // Obtener el word_index del tokenizer

// Cargar el modelo desde la memoria
const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson)); // Cargar el modelo desde el objeto en memoria
console.log(model.summary());

// Función para convertir texto en secuencias de índices
function textsToSequences(texts) {
    return texts.map(text => {
        const words = text.toLowerCase().trim().split(" ");
        return words.map(word => wordIndex[word] || 0); // Mapea cada palabra al índice correspondiente
    });
}

// Función para rellenar o truncar las secuencias
function padSequences(sequences, maxLength, paddingType = 'pre', truncatingType = 'pre', paddingValue = 0) {
    return sequences.map(seq => {
        if (seq.length > maxLength) {
            // Si la secuencia es más larga que maxLength, se trunca
            if (truncatingType === 'pre') {
                seq = seq.slice(seq.length - maxLength);
            } else {
                seq = seq.slice(0, maxLength);
            }
        }

        if (seq.length < maxLength) {
            // Si la secuencia es más corta que maxLength, se rellena
            const paddingLength = maxLength - seq.length;
            const paddingArray = new Array(paddingLength).fill(paddingValue);

            if (paddingType === 'pre') {
                seq = [...paddingArray, ...seq];
            } else {
                seq = [...seq, ...paddingArray];
            }
        }

        return seq;
    });
}

// Crear interfaz para capturar la entrada del usuario
const r1 = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const askQuestion = (question) => {
    return new Promise((resolve) => {
        r1.question(question, (answer) => {
            resolve(answer);
        });
    });
}

const main = async () => {
    let user_input = await askQuestion("Ingrese una frase: ");
    user_input = user_input.toLowerCase();

    // Convertir la entrada del usuario en secuencias de índices
    let sequences = textsToSequences([user_input]);
    console.log("Tokenized sequences:", sequences);

    // Aplicar padding a las secuencias
    sequences = padSequences(sequences, 5, 'pre', 'pre', 0);
    console.log("Padded sequences:", sequences);

    // Convertir las secuencias a un tensor
    const tensorInput = tf.tensor2d(sequences);
    console.log("Tensor input:", tensorInput);

    // Realizar una predicción
    const prediction = model.predict(tensorInput);

      // Imprimir el resultado de la predicción
    console.log("Predicción para el índice:", prediction.print());
}

main();