
require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs');


async function main() {

  const model = await use.loadQnA();

   // Base de conocimiento
   const responses = [
    'I am feeling great!',
    'The capital of China is Beijing.',
    'You have five fingers on your hand.',
    'The sky is blue on a clear day.',
    'Water boils at 100 degrees Celsius.',
    'Me llamo Vernik',
    'Tengo 27 anios'
  ];


  // Pregunta que deseas responder
  const question = 'Cuantos anios tienes?';

  // Crear input con la consulta y respuestas
  const input = {
    queries: [question],
    responses: responses
  };

  // Obtener las embeddings
  const embeddings = model.embed(input);

  // Calcular las similitudes entre la consulta y las respuestas
  const scores = tf.matMul(
    embeddings.queryEmbedding,
    embeddings.responseEmbedding,
    false,
    true
  ).arraySync();

  // Seleccionar la respuesta más relevante
  const bestMatchIndex = scores[0].indexOf(Math.max(...scores[0]));
  const bestResponse = responses[bestMatchIndex];

  console.log('Question:', question);
  console.log('Best Response:', bestResponse);
  }


  main().catch(console.error);