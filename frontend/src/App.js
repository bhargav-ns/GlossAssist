import logo from './logo.svg';
import './App.css';

import { useState } from 'react';

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async() => {
    if(!input.trim()) return;
    const userMessage = {role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    console.log(messages);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/message`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text:userMessage.text})
      });

      const data = await response.json();
      setMessages((prev) => [...prev, {role: "bot", text: data.text}]);
    }
    catch(error){
      setMessages((prev) => [...prev, {role: "bot", text: "error reaching server"}]);
      console.error(error);
    }
    finally{
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  }

  return (
    <div style = {{maxWidth: 600, margin: "60px auto", padding: "0 20px" }}>
      <h1> Sample App </h1>
      
      <div style = {{margin: "20px 0", minHeight: 100}}>
        {messages.length === 0 && <p>No messages yet.</p>}
        {messages.map((msg, i) => (
          <p key = {i}>
            <strong> {msg.role === "user"? "You" : "Bot"}</strong> {msg.text}
          </p>
        ))}
        {loading && <p> Loading... </p>}
      </div>

      <input 
        type = "text"
        value = {input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder='Type a message here...'
        style = {{width: "80%", marginRight: 8}}
      />
      <button onClick={sendMessage}>Send!</button>
    </div>
  )
}

export default App;
