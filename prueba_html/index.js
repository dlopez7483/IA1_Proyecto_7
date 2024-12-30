// Asegúrate de incluir TensorFlow.js si ejecutas en un navegador
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

async function loadModelAndTokenizer() {
    try {
        // Cargar el modelo y los pesos
        const model = await tf.loadLayersModel('model.json');
        console.log('Modelo cargado correctamente.');

        // Cargar el archivo tokenizer.json
        const response = await fetch('tokenizer.json');
        const tokenizer = await response.json();
        console.log('Tokenizador cargado correctamente.');

        return { model, tokenizer };
    } catch (error) {
        console.error('Error al cargar el modelo o el tokenizador:', error);
        throw error;
    }
}

function tokenizeInput(input, tokenizer, sequenceLength = 5) {
    // Convertir texto a minúsculas y dividir en palabras
    const words = input.toLowerCase().split(' ');

    // Mapear palabras a índices utilizando el tokenizador
    const indices = words.map(word => tokenizer.word_index[word] || 0);

    // Padding o trimming para asegurar que tenga la longitud requerida
    const paddedIndices = Array(sequenceLength).fill(0);
    for (let i = 0; i < Math.min(indices.length, sequenceLength); i++) {
        paddedIndices[i] = indices[i];
    }

    return paddedIndices;
}

async function testModel(inputText) {
    try {
        // Cargar modelo y tokenizador
        const { model, tokenizer } = await loadModelAndTokenizer();

        // Tokenizar la entrada
        const tokenizedInput = tokenizeInput(inputText, tokenizer);
        console.log('Entrada tokenizada:', tokenizedInput);

        // Convertir a tensor
        const inputTensor = tf.tensor([tokenizedInput]);
        console.log('Tensor de entrada:', inputTensor.toString());

        // Realizar la predicción
        const prediction = model.predict(inputTensor);

        // Convertir la salida a un arreglo normal y mostrar
        const output = prediction.arraySync()[0];
        console.log('Predicción del modelo:', output);
    } catch (error) {
        console.error('Error durante la predicción:', error);
    }
}

// Probar con una entrada
testModel('what is today').catch(console.error);