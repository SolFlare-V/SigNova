# Architecture Documentation

## System Overview

SignLang AI follows a **Layered Architecture** with **MVC + Service Pattern**, designed for modularity, testability, and extensibility.

## Layer Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                      │
│  React 18 + Vite + Tailwind CSS + Framer Motion          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Landing  │ │  Auth    │ │Dashboard │ │ Recogn.  │    │
│  │  Page    │ │  Pages   │ │  Page    │ │  Page    │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
├──────────────────────────────────────────────────────────┤
│                  APPLICATION LAYER                        │
│  FastAPI Controllers (Routers)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  Auth    │ │  Users   │ │ Gesture  │ │ Translate│    │
│  │ Router   │ │ Router   │ │ Router   │ │ Router   │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
├──────────────────────────────────────────────────────────┤
│                   SERVICE LAYER                           │
│  OOP Service Classes (Business Logic)                    │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────┐     │
│  │AuthService  │ │GestureService│ │TranslationSvc │     │
│  │UserService  │ │ModelLoader   │ │               │     │
│  └─────────────┘ └──────────────┘ └───────────────┘     │
├──────────────────────────────────────────────────────────┤
│                AI / ML SERVICE LAYER                      │
│  MediaPipe Hands → Feature Extraction → ML Classifier    │
│  Pipeline: Frame → Landmarks → Vector → Prediction       │
├──────────────────────────────────────────────────────────┤
│                 TRANSLATION LAYER                         │
│  deep-translator (Google Translate) + Fallback Dict      │
│  Languages: English → Tamil, Hindi                       │
├──────────────────────────────────────────────────────────┤
│                    DATA LAYER                             │
│  SQLAlchemy ORM + SQLite                                 │
│  Tables: users, session_logs                             │
└──────────────────────────────────────────────────────────┘
```

## Design Patterns Used

### MVC (Model-View-Controller)
- **Model**: SQLAlchemy ORM models (`User`, `SessionLog`)
- **View**: React components (pages and UI)
- **Controller**: FastAPI routers handling HTTP requests

### Service Pattern
Each domain has a dedicated service class:
- `AuthService` — JWT authentication, password hashing
- `UserService` — Profile CRUD operations
- `GestureService` — Image processing, landmark extraction, prediction
- `TranslationService` — Multilingual text translation
- `ModelLoader` — Singleton ML model management

### Singleton Pattern
`ModelLoader` uses singleton to ensure model is loaded once at startup.

### Repository Pattern
Database operations are encapsulated within service classes, abstracting SQLAlchemy queries from routers.

## Database Schema

### users
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| username | VARCHAR(50) | Unique, indexed |
| email | VARCHAR(100) | Unique, indexed |
| full_name | VARCHAR(100) | Display name |
| hashed_password | VARCHAR(255) | Bcrypt hash |
| role | VARCHAR(20) | user/admin |
| preferred_language | VARCHAR(10) | en/ta/hi |
| theme | VARCHAR(20) | dark/midnight |
| is_active | BOOLEAN | Account status |
| created_at | DATETIME | Auto-set |
| updated_at | DATETIME | Auto-update |

### session_logs
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| user_id | INTEGER FK | References users |
| gesture_detected | VARCHAR(100) | Detected gesture |
| confidence | FLOAT | ML confidence |
| translated_text | TEXT | Translation output |
| target_language | VARCHAR(10) | Target lang code |
| duration_ms | INTEGER | Processing time |
| created_at | DATETIME | Auto-set |

## Future Extensibility

The modular design supports:
1. **CNN/LSTM Models** — Swap `ModelLoader` to load deep learning models
2. **Transformer Translation** — Replace `TranslationService` backend
3. **Speech Synthesis** — Add `SpeechService` as new service layer
4. **Edge AI** — Export model to TensorFlow.js for client-side inference
5. **Multi-Camera** — Extend `GestureService` for multiple camera inputs
6. **Mobile App** — API-first design enables React Native integration
7. **Dataset Collection** — Add data collection endpoints to `GestureRouter`
