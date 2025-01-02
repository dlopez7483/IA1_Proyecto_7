#from chat import Chatbot
from chat import responder
# Crear una instancia del Chatbot
##chatbot = Chatbot('intents.json', 'model.h5')

# Interactuar con el chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: ¡Adiós!")
        break
    ##response = chatbot.responder(user_input)
    response = responder(user_input)
    print(f"Chatbot: {response}")
    