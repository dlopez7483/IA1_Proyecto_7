import React, { useState, useEffect, useRef } from "react";
import Mensaje from "./Mensaje";
import { getResponse } from "../models/modelo_js/modelHandler"; // Importar la función

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false); // Estado para manejar la carga
  const chatEndRef = useRef(null); // Para el scroll automático

  // Hacer scroll hacia abajo cuando se agregan mensajes
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom(); // Scroll automático
  }, [messages]);

  // Manejar el envío de mensajes
  const handleSendMessage = async () => {
    if (input.trim() === "") return; // No enviar mensajes vacíos

    const timestamp = new Date().toLocaleString();
    const userMessage = { text: input, sender: "user", timestamp };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    setIsLoading(true);

    try {
      // Llamar a la función getResponse para obtener la respuesta del modelo
      const botResponse = await getResponse(input);
      const botMessage = { text: botResponse, sender: "bot", timestamp };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      const errorMessage = {
        text: "Ocurrió un error al procesar tu mensaje. Intenta nuevamente.",
        sender: "bot",
        timestamp,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      console.error("Error al obtener la respuesta:", error);
    }

    setIsLoading(false); // Finalizar el estado de carga
    setInput(""); // Limpiar el campo de entrada
  };

  // Enviar mensaje al presionar Enter
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">Modelo IA Fase 1</div>
      <div className="chat-messages">
        {isLoading && <p>Cargando respuesta...</p>}
        {messages.map((msg, index) => (
          <Mensaje
            key={index}
            text={msg.text}
            sender={msg.sender}
            timestamp={msg.timestamp}
          />
        ))}
        <div ref={chatEndRef} /> {/* Bandera para el scroll automático */}
      </div>
      <div className="input-group">
        <input
          type="text"
          placeholder="Escribe un mensaje..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button onClick={handleSendMessage}>Enviar</button>
      </div>
    </div>
  );
};

export default Chat;
