
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
    'Tengo 27 anios',
    'Soy de Guatemala',
    'Mi color favorito es el azul',
    'Mi comida favorita es la pizza',
    'Mi pelicula favorita es Star Wars',
    'Mi serie favorita es Breaking Bad',
    'Mi libro favorito es El Alquimista',
    'Mi deporte favorito es el futbol',
    'Mi equipo favorito es el Barcelona',
    'Mi animal favorito es el perro',
    'Mi pasatiempo favorito es leer',
    'Mi hobby favorito es cocinar',
    'Mi lugar favorito es la playa',
    'Mi cancion favorita es come as you are',
    'Mi artista favorito es Nirvana',
    'Mi grupo favorito es Nirvana',
    'Mi genero favorito es el rock',
    'Mi estilo favorito es el grunge',
    'Mi marca favorita es Nike',
    'Mi marca favorita es Adidas',
    'Mi marca favorita es Puma',
    'Mi marca favorita es Vans',
    "Hola, ¿cómo puedo ayudarte hoy?",
    "Lo siento, no entiendo lo que estás diciendo.",
    "Estoy aquí para responder tus preguntas.",
    "Eso suena interesante, ¿quieres saber más sobre eso?",
    "¿En qué más puedo ayudarte?",
    "El lenguaje de programación más utilizado es Python.",
    "La inteligencia artificial está cambiando el mundo rápidamente.",
    "El sistema operativo más popular es Windows.",
    "Los servidores en la nube son una parte fundamental de la tecnología moderna.",
    "La programación orientada a objetos es una de las principales metodologías de desarrollo.",
    "El monte Everest es la montaña más alta del mundo.",
    "Francia es famosa por la Torre Eiffel.",
    "La capital de España es Madrid.",
    "Brasil es conocido por sus hermosas playas y la selva amazónica.",
    "México tiene una rica historia y cultura.",
    "Albert Einstein es conocido por la teoría de la relatividad.",
    "El sol es una estrella que da luz y calor a la Tierra.",
    "El agua es esencial para la vida en la Tierra.",
    "La Mona Lisa es una de las obras más famosas de Leonardo da Vinci.",
    "La Revolución Francesa ocurrió a finales del siglo XVIII.",
    "Mi comida favorita es el sushi.",
    "Me encanta viajar por el mundo.",
    "Disfruto mucho de la música clásica.",
    "Mi deporte favorito es el tenis.",
    "Soy un fanático del cine de terror.",
    "Soy como un robot, pero un poco más divertido.",
    "¡Estoy tan feliz de hablar contigo hoy!",
    "¡Me encanta aprender cosas nuevas cada día!",
    "Soy una inteligencia artificial, pero también tengo sentido del humor.",
    "El cielo es el límite, ¡vamos a explorar todo el conocimiento!",
    "¡Hola! ¿En qué puedo ayudarte hoy?",
    "¡Qué bueno que me contactes, cuéntame qué necesitas saber!",
    "No estoy seguro de eso, pero puedo investigar más información.",
    "¿Te gustaría saber algo más sobre este tema?",
    "No entiendo bien lo que estás diciendo, ¿puedes repetirlo?",
    "Lo siento, no puedo ayudarte con eso en este momento.",
    "No tengo información suficiente sobre ese tema.",
    "No estoy seguro de la respuesta, pero puedo buscar más detalles.",
    "Esa no es una pregunta que pueda responder ahora."
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