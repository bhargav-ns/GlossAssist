# Full-Stack ML App

A full-stack application with a React frontend, Flask backend, and a PyTorch character-level language model — all containerised with Docker and deployed on AWS EC2.

---

## Stack

- **Frontend**: React (port 3000)
- **Backend**: Flask (port 5001)
- **Model**: PyTorch LSTM (character-level language model)
- **Containerisation**: Docker + Docker Compose
- **Hosting**: AWS EC2 (Ubuntu 24.04)

---

## Project Structure

```
project/
├── frontend/
│   ├── Dockerfile
│   ├── .env
│   └── src/
│       └── App.jsx
├── backend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── model/
│   ├── Dockerfile
│   ├── app.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── requirements.txt
│   └── data/
│       └── input.txt
└── docker-compose.yml
```

---

## Local Development

### Prerequisites

- Docker + Docker Compose
- Node.js 18+
- Python 3.11+

### 1. Clone the repo

```bash
git clone https://github.com/you/your-repo.git
cd your-repo
```

### 2. Set up environment variables

**`frontend/.env`**
```
REACT_APP_API_URL=http://localhost:5001
REACT_APP_API_KEY=your-secret-key-here
```

**`model/.env`** (used when running without Docker)
```
API_KEY=your-secret-key-here
```

### 3. Train the model

Add a plain text file to `model/data/input.txt`, then:

```bash
cd model
pip install torch
python train.py
```

This produces `model.pt` in the `model/` directory. Training takes a few minutes on CPU.

### 4. Run with Docker Compose

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:5001

---

## API

### `GET /`
Health check.

**Response**
```
Ok
```

### `POST /predict`
Generate text from a seed string.

**Headers**
```
Content-Type: application/json
X-API-Key: your-secret-key-here
```

**Request body**
```json
{
  "text": "The ",
  "length": 200
}
```

**Response**
```json
{
  "text": "The generated text continues here..."
}
```

**Error responses**
- `400` — no seed text provided
- `401` — missing or invalid API key

---

## AWS EC2 Deployment

### 1. Launch an EC2 instance

- AMI: Ubuntu 24.04 LTS
- Instance type: t2.micro
- Create and download a `.pem` key pair

### 2. Configure security group inbound rules

| Port | Source    |
|------|-----------|
| 22   | Your IP   |
| 3000 | 0.0.0.0/0 |
| 5001 | 0.0.0.0/0 |

### 3. SSH into the instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 4. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
newgrp docker
```

### 5. Deploy

```bash
git clone https://github.com/you/your-repo.git
cd your-repo
```

Update `frontend/.env` to point to the EC2 public IP:
```
REACT_APP_API_URL=http://your-ec2-public-ip:5001
```

Then run:
```bash
docker-compose up --build -d
```

Visit `http://your-ec2-public-ip:3000`.

---

## Common Gotchas

- Flask must run with `host="0.0.0.0"` or it won't be reachable outside its container
- `REACT_APP_API_URL` uses `localhost` locally and the EC2 public IP when deployed — update `.env` before deploying
- `model.pt` must be present in the `model/` directory before building the Docker image
- EC2 public IP changes every time the instance stops and starts — use an Elastic IP for a permanent address
- Always use `http://` explicitly in the browser, not `https://`
- Add `.env` to `.gitignore` so API keys are never pushed to Git

---

## Security Notes

- The API key is passed via the `X-API-Key` request header, never in the URL
- Environment variables are used for all secrets — nothing is hardcoded
- `REACT_APP_API_KEY` is baked into the frontend bundle at build time — in production, proxy requests through your own backend so the key never reaches the browser

---

## Next Steps

This project is a foundation for deploying a LaBSE-based model. To extend it:

1. Replace the `CharLSTM` in `model/` with your LaBSE-based PyTorch model
2. Update the `predict()` function to match your model's input/output format
3. Add nginx as a reverse proxy to consolidate ports 3000 and 5001 behind port 80
4. Use AWS ECS + ECR instead of a bare EC2 instance for production container hosting
5. Set up a CI/CD pipeline with GitHub Actions to automate deployment on push