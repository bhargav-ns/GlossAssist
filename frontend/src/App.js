import logo from "./logo.svg";
import "./App.css";

import { useState, useRef } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [trainingFile, setTrainingFile] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

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

  const DEFAULT_PARAMS = {
  model: "LSTM",
  epochs: 20,
  lr: 0.01,
  batch_size: 64,
  emb_dim: 64,
  hidden_dim: 128,
  num_layers: 2,
  seq_len: 100,
  tokenization_type: "char",
};

const [params, setParams] = useState(DEFAULT_PARAMS);
const [trainStatus, setTrainStatus] = useState(null);
const pollRef = useRef(null);

const handleParamChange = (key, value) => {
  setParams(prev => ({ ...prev, [key]: value }));
};

const handleTrain = async () => {
  setTrainingLoading(true);
  setTrainStatus(null);

  try {
    const response = await fetch(`${process.env.REACT_APP_API_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const data = await response.json();
    if (data.error) { alert(data.error); return; }

    // Start polling
    pollRef.current = setInterval(async () => {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/train/status`);
      const status = await res.json();
      setTrainStatus(status);
      if (!status.running) {
        clearInterval(pollRef.current);
        setTrainingLoading(false);
        if (status.done) {
          setFileUploaded(false);
          setTrainingFile(null);
        }
      }
    }, 1000);

  } catch (error) {
    alert("Error starting training: " + error.message);
    setTrainingLoading(false);
  }
};

  return (
    <div className="app">
      <header className="app-header">
        <h1>Custom Chat Agent</h1>
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
          </div>
          {trainingFile && (
            <p className="file-name">Selected: {trainingFile.name}</p>
          )}
          {fileUploaded && (
            <p className="status-pill status-ready">
              ✓ File uploaded, ready to train
            </p>
          )}

          <div className="hyperparam-grid">
            <div className="param-group">
              <label>Model Type</label>
              <select
                value={params.model}
                onChange={(e) =>
                  handleParamChange("model", e.target.value)
                }
              >
                <option value="lstm">LSTM</option>
                <option value="rnn">RNN</option>
              </select>
            </div>
            <div className="param-group">
              <label>Epochs</label>
              <input
                type="number"
                value={params.epochs}
                min={1}
                onChange={(e) =>
                  handleParamChange("epochs", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Learning rate</label>
              <input
                type="number"
                value={params.lr}
                step={0.001}
                min={0.0001}
                onChange={(e) =>
                  handleParamChange("lr", parseFloat(e.target.value))
                }
              />
            </div>
            

            <div className="param-group">
              <label>Batch size</label>
              <input
                type="number"
                value={params.batch_size}
                min={1}
                onChange={(e) =>
                  handleParamChange("batch_size", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Embedding dim</label>
              <input
                type="number"
                value={params.emb_dim}
                min={8}
                onChange={(e) =>
                  handleParamChange("emb_dim", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Hidden dim</label>
              <input
                type="number"
                value={params.hidden_dim}
                min={8}
                onChange={(e) =>
                  handleParamChange("hidden_dim", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Layers</label>
              <input
                type="number"
                value={params.num_layers}
                min={1}
                max={8}
                onChange={(e) =>
                  handleParamChange("num_layers", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Seq length</label>
              <input
                type="number"
                value={params.seq_len}
                min={10}
                onChange={(e) =>
                  handleParamChange("seq_len", parseInt(e.target.value))
                }
              />
            </div>
            <div className="param-group">
              <label>Tokenization</label>
              <select
                value={params.tokenization_type}
                onChange={(e) =>
                  handleParamChange("tokenization_type", e.target.value)
                }
              >
                <option value="char">Char</option>
                <option value="whitespace">Whitespace</option>
                <option value="bpe">BPE</option>
              </select>
            </div>
          </div>

          {trainStatus?.running && (
            <div className="train-progress">
              <div className="progress-header">
                <span>
                  Epoch {trainStatus.epoch} / {trainStatus.total_epochs}
                </span>
                <span>Loss: {trainStatus.loss ?? "—"}</span>
              </div>
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${(trainStatus.epoch / trainStatus.total_epochs) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}

          {trainStatus?.done && (
            <p className="status-pill status-ready">✓ Training complete</p>
          )}
          {trainStatus?.error && (
            <p className="status-pill status-error">✗ {trainStatus.error}</p>
          )}

          <button
            className="btn"
            onClick={handleTrain}
            disabled={!fileUploaded || trainingLoading}
            style={{ marginTop: 16 }}
          >
            {trainingLoading ? "Training..." : "Start training"}
          </button>
        </div>
        
      </div>
    </div>
  );
}

export default App;
