import logo from "./logo.svg";
import "./App.css";

import { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [trainingFile, setTrainingFile] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    console.log(messages);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userMessage.text }),
      });

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "bot", text: data.text }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "error reaching server" },
      ]);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadFile = async () => {
    if (!trainingFile) return;

    setUploadLoading(true);
    const formData = new FormData();
    formData.append("file", trainingFile);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      alert("File uploaded successfully!");
      setFileUploaded(true);
    } catch (error) {
      alert("Error uploading file: " + error.message);
      console.error(error);
    } finally {
      setUploadLoading(false);
    }
  };

  const handleTrain = async () => {
    setTrainingLoading(true);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();
      alert("Training started!");
      setTrainingFile(null);
      setFileUploaded(false);
    } catch (error) {
      alert("Error starting training: " + error.message);
      console.error(error);
    } finally {
      setTrainingLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Sample App</h1>
      </header>

      <div className="chat-section">
        <div className="chat-window">
          {messages.length === 0 && (
            <p className="empty-state">No messages yet.</p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.role === "user" ? "message-user" : "message-bot"}`}
            >
              <div className="message-label">
                {msg.role === "user" ? "You" : "Bot"}
              </div>
              {msg.text}
            </div>
          ))}
          {loading && <p className="loading">Thinking...</p>}
        </div>

        <div className="input-row">
          <input
            className="chat-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
          />
          <button className="btn" onClick={sendMessage}>
            Send
          </button>
        </div>
      </div>

      <hr className="divider" />

      <div className="train-section">
        <div className="section-label">Train model</div>
        <div className="file-row">
          <label className="file-input-label">
            <input
              type="file"
              onChange={(e) => setTrainingFile(e.target.files[0])}
              accept=".txt"
              hidden
            />
            Choose .txt file
          </label>
          <button
            className="btn btn-secondary"
            onClick={handleUploadFile}
            disabled={!trainingFile || uploadLoading}
          >
            {uploadLoading ? "Uploading..." : "Upload file"}
          </button>
          <button
            className="btn"
            onClick={handleTrain}
            disabled={!fileUploaded || trainingLoading}
          >
            {trainingLoading ? "Training..." : "Start training"}
          </button>
        </div>
        {trainingFile && (
          <p className="file-name">Selected: {trainingFile.name}</p>
        )}
        {fileUploaded && (
          <p className="status-pill status-ready">
            File uploaded, ready to train
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
