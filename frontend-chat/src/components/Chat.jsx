import React, { useState, useRef, useEffect } from "react";
import Mensaje from "./Mensaje";
import {loadModel, getResponse} from '../utils/modelo'

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const chatEndRef = useRef(null); // el useRef se está utilziando para el scroll automático
  
    useEffect(() => {
      const initializeModel = async () => {
        await loadModel();
      };
      initializeModel();
    }, []);

    // el scroll se maneja ahora para que vaya bajando automáticamente según ingresan mensajes:
    const scrollToBottom = () => {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
  
    useEffect(() => {
      scrollToBottom(); 
    }, [messages]);
  
    const handleSendMessage = async () => {
      if (input.trim() === "") return;
  
      const timestamp = new Date().toLocaleString();
  
      const userMessage = { text: input, sender: "user", timestamp };
<<<<<<< HEAD
      setMessages((prevMessages) => [...prevMessages, userMessage]);
=======
      const botMessage = { text: "Esta es una respuestaaa", sender: "bot", timestamp };
>>>>>>> ramas/201907483
  
      try {
        const botResponse = await getResponse(input);
        const botMessage = { text: botResponse, sender: "bot", timestamp };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      } catch (error) {
        const errorMessage = { text: "Ocurrió un error. Envía un mensaje nuevamente...", sender: "bot", timestamp };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      }
  
      setInput(""); 
    };
  
    const handleKeyPress = (event) => {
      if (event.key === "Enter") {
        handleSendMessage();
      }
    };
  
    return (
      <div className="chat-container">
        <div className="chat-header">Modelo IA Fase 1</div>
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <Mensaje key={index} text={msg.text} sender={msg.sender} timestamp={msg.timestamp} />
          ))}
          <div ref={chatEndRef} /> {/* bandera para el scroll automático */}
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