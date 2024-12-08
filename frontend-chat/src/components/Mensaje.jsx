import React from "react";

const Mensaje = ({ text, sender, timestamp }) => {
    const isUser = sender === "user";
    return (
      <div className={`chat-message ${isUser ? "user" : "bot"}`}>
        <div className="message-bubble">
          <p>{text}</p>
          <span className="message-timestamp">{timestamp}</span>
        </div>
      </div>
    );
};

export default Mensaje;
