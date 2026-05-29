# 🏥 MedAI Nexus

<div align="center">

![MedAI Nexus Banner](https://img.shields.io/badge/MedAI-Nexus-00ccf0?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJhMTAgMTAgMCAxIDAgMCAyMEExMCAxMCAwIDAgMCAxMiAyeiIgZmlsbD0iIzAwY2NmMCIvPjwvc3ZnPg==)

**Next-Generation AI Healthcare Intelligence Platform**

*Powered by LLMs · RAG Pipeline · Autonomous Multi-Agent AI*

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs)](https://nextjs.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/yourusername/medai-nexus/ci-cd.yml?label=CI%2FCD)](https://github.com/yourusername/medai-nexus/actions)

[**Live Demo**](https://medai-nexus.vercel.app) · [**API Docs**](https://api.medai-nexus.dev/api/docs) · [**Architecture**](#architecture)

</div>

---

## 🎯 Overview

**MedAI Nexus** is a production-grade AI healthcare intelligence platform that combines cutting-edge **Large Language Models**, **Retrieval-Augmented Generation (RAG)**, and **Autonomous Multi-Agent AI** to deliver hospital-grade medical insights to anyone, anywhere.

> Built as a final-year major project demonstrating advanced AI/ML engineering, full-stack development, and DevOps — suitable for **hackathons**, **internship applications**, and **GitHub portfolio**.

---

## ✨ Key Features

| Feature | Technology | Status |
|---------|-----------|--------|
| 🤖 AI Doctor Chat | GPT-4 + LangGraph + RAG | ✅ Live |
| 📄 Medical Report Analysis | OCR + PyMuPDF + ChromaDB | ✅ Live |
| 💊 Prescription Analyzer | Tesseract + Drug DB | ✅ Live |
| 🚨 Emergency Detection Agent | Autonomous AI Agent | ✅ Live |
| 📊 Health Analytics Dashboard | Predictive ML Models | ✅ Live |
| 🏥 Nearby Hospital Finder | Google Maps API | ✅ Live |
| ⏰ Medicine Reminders | Celery + Redis | ✅ Live |
| 🔐 JWT + OAuth Authentication | FastAPI + JWT | ✅ Live |
| 📡 Real-time Streaming Chat | WebSockets | ✅ Live |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedAI Nexus Platform                      │
├─────────────────┬───────────────────┬───────────────────────────┤
│   Frontend      │    Backend        │    AI/ML Layer             │
│                 │                   │                            │
│  Next.js 14    │  FastAPI          │  LangGraph Orchestrator    │
│  TypeScript     │  WebSockets       │  ┌─────────────────────┐  │
│  Three.js       │  REST API v1      │  │  OCR Agent          │  │
│  Tailwind CSS   │                   │  │  Medical Agent      │  │
│  Framer Motion  │  PostgreSQL       │  │  RAG Agent          │  │
│  ShadCN UI      │  Redis Cache      │  │  Emergency Agent    │  │
│                 │  ChromaDB         │  │  Recommendation     │  │
│                 │                   │  │  Reminder Agent     │  │
│                 │  Docker +         │  └─────────────────────┘  │
│                 │  Nginx            │  GPT-4 / Anthropic Claude  │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### RAG Pipeline Flow
```
Upload → OCR Extraction → Text Chunking → Embedding Generation
      → ChromaDB Upsert → Semantic Search → Cross-Encoder Reranking
      → LLM Context Injection → Structured Medical Response
```

### Multi-Agent Workflow
```
User Input
    │
    ▼
Emergency Triage Agent ──(critical)──► Emergency Protocol
    │ (normal)
    ▼
RAG Retrieval Agent (ChromaDB semantic search)
    │
    ▼
Medical Analysis Agent (GPT-4 with medical context)
    │
    ▼
Recommendation Agent → Health Score + Action Items
    │
    ▼
Reminder Agent (schedule follow-ups via Celery)
```

---

## 🛠️ Tech Stack

### Frontend
```
Next.js 14 (App Router)    TypeScript         Tailwind CSS
Three.js + R3F             Framer Motion      GSAP
ShadCN/UI                  Recharts           Zustand
```

### Backend
```
FastAPI 0.115   Python 3.12     AsyncPG (async PostgreSQL)
Redis 7         WebSockets      Celery + Flower
SQLAlchemy 2    Alembic         Pydantic v2
```

### AI / ML
```
LangChain 0.3   LangGraph       OpenAI GPT-4o
Anthropic Claude ChromaDB       Sentence Transformers
Tesseract OCR   PyMuPDF         OpenCV
HuggingFace     Cross-Encoder   RAG Pipeline
```

### DevOps
```
Docker Compose  GitHub Actions  Nginx
Vercel          Render/AWS      PostgreSQL 16
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key (or Anthropic API Key)
- Node.js 20+ and Python 3.12+

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medai-nexus.git
cd medai-nexus
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

### 3. Launch with Docker
```bash
docker compose up --build
```

### 4. Access Platform
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/api/docs |
| ChromaDB | http://localhost:8001 |

---

## 🔧 Manual Setup (Without Docker)

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Setup database
alembic upgrade head

# Start server
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
medai-nexus/
├── frontend/                    # Next.js 14 App
│   ├── app/                     # App Router pages
│   │   ├── (auth)/              # Login, Register
│   │   ├── chat/                # AI Doctor Chat
│   │   ├── reports/             # Report Upload
│   │   ├── analytics/           # Health Dashboard
│   │   ├── emergency/           # Emergency Detection
│   │   └── prescriptions/       # Rx Analyzer
│   ├── components/
│   │   ├── ui/                  # ShadCN components
│   │   ├── three/               # 3D visualizations
│   │   ├── chat/                # Chat UI components
│   │   └── charts/              # Analytics charts
│   ├── lib/
│   │   ├── api/                 # API client
│   │   └── stores/              # Zustand state
│   └── public/shaders/          # GLSL shaders
│
├── backend/                     # FastAPI Application
│   ├── main.py                  # App entrypoint
│   ├── core/
│   │   ├── config.py            # Settings
│   │   ├── database.py          # Async DB setup
│   │   └── redis_client.py      # Redis connection
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph workflow
│   │   ├── ocr_agent.py         # OCR extraction
│   │   ├── rag_agent.py         # RAG retrieval
│   │   ├── medical_agent.py     # Medical analysis
│   │   ├── emergency_agent.py   # Emergency detection
│   │   ├── recommendation_agent.py
│   │   └── reminder_agent.py
│   ├── routes/
│   │   ├── auth.py              # JWT + OAuth
│   │   ├── chat.py              # Chat + WebSocket
│   │   ├── reports.py           # File upload + RAG
│   │   ├── emergency.py         # Emergency API
│   │   ├── analytics.py         # Health insights
│   │   ├── reminders.py         # Reminders
│   │   └── hospitals.py         # Maps API
│   ├── models/
│   │   └── models.py            # SQLAlchemy ORM
│   ├── middleware/
│   │   ├── auth.py              # JWT validation
│   │   └── rate_limit.py        # Rate limiting
│   ├── migrations/              # Alembic migrations
│   ├── tests/                   # PyTest test suite
│   ├── requirements.txt
│   └── Dockerfile
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions
│
├── nginx/
│   └── nginx.conf               # Reverse proxy config
│
├── docker-compose.yml           # Full stack orchestration
├── .env.example                 # Environment template
└── README.md
```

---

## 🌐 API Endpoints

### Authentication
```
POST   /api/v1/auth/register        Register new user
POST   /api/v1/auth/login           JWT login
POST   /api/v1/auth/oauth/google    Google OAuth
GET    /api/v1/auth/me              Current user profile
POST   /api/v1/auth/refresh         Refresh token
```

### AI Chat
```
POST   /api/v1/chat/message         Send chat message
GET    /api/v1/chat/conversations   List conversations
GET    /api/v1/chat/conversations/{id}/messages
WS     /api/v1/chat/ws/{conv_id}    Streaming chat
```

### Medical Reports
```
POST   /api/v1/reports/upload       Upload + analyze report
GET    /api/v1/reports              List user reports
GET    /api/v1/reports/{id}         Report details + insights
DELETE /api/v1/reports/{id}         Delete report
```

### Emergency
```
POST   /api/v1/emergency/assess     Symptom assessment
GET    /api/v1/emergency/logs       Emergency history
POST   /api/v1/emergency/alert      Trigger alert protocol
```

### Analytics
```
GET    /api/v1/analytics/dashboard  Health dashboard data
GET    /api/v1/analytics/trends     Health trends over time
GET    /api/v1/analytics/risks      AI risk predictions
GET    /api/v1/analytics/score      Overall health score
```

---

## 🔑 Environment Variables

```bash
# .env.example

# Core
SECRET_KEY=your-super-secret-key-minimum-32-chars
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://medai:password@localhost:5432/medai_nexus
REDIS_URL=redis://localhost:6379/0

# AI
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Vector DB
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Maps
GOOGLE_MAPS_KEY=AIza...

# Email (for reminders)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=your-app-password
```

---

## 🧪 Testing

```bash
# Backend
cd backend
pytest tests/ -v --cov=. --cov-report=html
open htmlcov/index.html

# Frontend
cd frontend
npm test
npm run e2e        # Playwright E2E tests
```

---

## 📈 Resume Highlights

This project demonstrates:

- **LLM Engineering** — Prompt engineering, streaming, token management
- **RAG Implementation** — End-to-end retrieval-augmented generation pipeline
- **Multi-Agent AI** — LangGraph orchestrated autonomous agent workflows
- **Vector Databases** — ChromaDB with semantic search and reranking
- **OCR Pipelines** — Tesseract + OpenCV + PyMuPDF document processing
- **Full-Stack Development** — Next.js 14 + FastAPI production app
- **Real-time Systems** — WebSocket streaming for live AI responses
- **Scalable Architecture** — Microservices, async Python, Redis caching
- **DevOps/CI-CD** — Docker, GitHub Actions, automated deployment
- **Security** — JWT auth, rate limiting, HIPAA-aware design

---

## 🤝 Contributing

```bash
# Fork → Clone → Branch → Commit → PR
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) for LLM orchestration framework
- [OpenAI](https://openai.com) for GPT-4 and embeddings
- [ChromaDB](https://trychroma.com) for vector storage
- [FastAPI](https://fastapi.tiangolo.com) for the excellent async web framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for open-source OCR

---

<div align="center">

**Built with ❤️ for the future of healthcare**

⭐ Star this repo if it helped you!

</div>
