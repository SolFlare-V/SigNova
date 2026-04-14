# Setup & Deployment Guide

## Local Development Setup

### 1. Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Generate demo ML model (for testing without real data)
python ml/train_model.py --demo

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 2. Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at `http://localhost:5173`

### 3. Training a Real Model (Optional)

```bash
cd backend

# Collect training data via webcam
python ml/train_model.py --collect

# Train model from collected data
python ml/train_model.py --train
```

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

Access the app at `http://localhost`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| SECRET_KEY | (change me) | JWT signing secret |
| ALGORITHM | HS256 | JWT algorithm |
| ACCESS_TOKEN_EXPIRE_MINUTES | 60 | Token expiry |
| DATABASE_URL | sqlite:///./signlang.db | Database connection |
| CORS_ORIGINS | http://localhost:5173 | Allowed origins |
| MODEL_PATH | ml/models/gesture_model.pkl | ML model location |

## Production Checklist

- [ ] Change `SECRET_KEY` to a strong random value
- [ ] Switch SQLite to PostgreSQL for concurrent access
- [ ] Set `DEBUG=False`
- [ ] Use HTTPS with proper certificates
- [ ] Set up proper CORS origins
- [ ] Configure rate limiting
- [ ] Enable production logging
- [ ] Use Google Cloud Translation API instead of free tier
