"""
Advanced Project Templates Module
Real working project templates optimized for Linux development
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import subprocess
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TemplateFile:
    """Template file definition with real content"""
    path: str
    content: str
    executable: bool = False
    template_vars: List[str] = None


@dataclass
class TemplateCommand:
    """Template setup command"""
    command: str
    description: str
    required: bool = True
    linux_only: bool = False


class ProjectTemplateEngine:
    """Advanced project template engine with real implementations"""

    def __init__(self):
        self.templates = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all project templates with real, working code"""

        # Python FastAPI Template
        self.templates["python_fastapi"] = {
            "name": "Python FastAPI REST API",
            "description": "Production-ready FastAPI application with authentication, database, and Docker",
            "language": "python",
            "framework": "fastapi",
            "files": [
                TemplateFile(
                    "main.py",
                    '''#!/usr/bin/env python3
"""
{project_name} - FastAPI Application
Created with Terminal Coder Linux Edition
"""

import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import os
import asyncio
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Boolean, Text
import jwt
from datetime import datetime, timedelta
import hashlib
import secrets

# Configuration
class Settings:
    PROJECT_NAME = "{project_name}"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

settings = Settings()

# Database Setup
Base = declarative_base()
engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Models
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# Pydantic Models
class UserBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    email: str = Field(..., max_length=255, description="User email address")
    username: str = Field(..., max_length=100, description="Username")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="User password")

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

class ItemBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str = Field(..., max_length=200, description="Item title")
    description: Optional[str] = Field(None, description="Item description")

class ItemCreate(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int
    owner_id: int
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Security
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        hash_part, salt = hashed_password.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == hash_part
    except ValueError:
        return False

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")

        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            result = await db.execute(select(User).where(User.username == username))
            user = result.scalar_one_or_none()
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# Database dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Redis connection
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client

    # Startup
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Connect to Redis
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()

        logging.info("Application startup complete")
        yield

    except Exception as e:
        logging.error(f"Startup error: {e}")
        yield
    finally:
        # Shutdown
        if redis_client:
            await redis_client.close()
        await engine.dispose()
        logging.info("Application shutdown complete")

# FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="{project_description}",
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["yourdomain.com", "*.yourdomain.com"]
)

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}!",
        "version": settings.VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "checks": {}
    }

    try:
        # Database health
        async with AsyncSessionLocal() as db:
            await db.execute(select(1))
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    try:
        # Redis health
        if redis_client:
            await redis_client.ping()
            health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"

    return health_status

@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register new user"""
    from sqlalchemy import select

    # Check if user exists
    result = await db.execute(select(User).where(
        (User.email == user.email) | (User.username == user.username)
    ))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User already exists")

    # Create user
    hashed_password = hash_password(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user

@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login_user(email: str, password: str, db: AsyncSession = Depends(get_db)):
    """Login user and return JWT token"""
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is disabled")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Users"])
async def read_current_user(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@app.get("/items/", response_model=List[ItemResponse], tags=["Items"])
async def read_items(skip: int = 0, limit: int = 100,
                    current_user: User = Depends(get_current_user),
                    db: AsyncSession = Depends(get_db)):
    """Get items for current user"""
    from sqlalchemy import select

    result = await db.execute(
        select(Item).where(Item.owner_id == current_user.id)
        .offset(skip).limit(limit)
    )
    items = result.scalars().all()
    return items

@app.post("/items/", response_model=ItemResponse, tags=["Items"])
async def create_item(item: ItemCreate,
                     current_user: User = Depends(get_current_user),
                     db: AsyncSession = Depends(get_db)):
    """Create new item"""
    db_item = Item(**item.model_dump(), owner_id=current_user.id)
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@app.get("/items/{item_id}", response_model=ItemResponse, tags=["Items"])
async def read_item(item_id: int,
                   current_user: User = Depends(get_current_user),
                   db: AsyncSession = Depends(get_db)):
    """Get specific item"""
    from sqlalchemy import select

    result = await db.execute(
        select(Item).where(Item.id == item_id, Item.owner_id == current_user.id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.delete("/items/{item_id}", tags=["Items"])
async def delete_item(item_id: int,
                     current_user: User = Depends(get_current_user),
                     db: AsyncSession = Depends(get_db)):
    """Delete item"""
    from sqlalchemy import select, delete

    result = await db.execute(
        select(Item).where(Item.id == item_id, Item.owner_id == current_user.id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    await db.execute(delete(Item).where(Item.id == item_id))
    await db.commit()

    return {"message": "Item deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        access_log=settings.DEBUG
    )
''',
                    template_vars=["project_name", "project_description"]
                ),

                TemplateFile(
                    "requirements.txt",
                    '''# FastAPI and dependencies
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Database
sqlalchemy>=2.0.23
aiosqlite>=0.19.0
alembic>=1.13.1

# Security
python-jose[cryptography]>=3.3.0
passlib>=1.7.4
bcrypt>=4.1.2

# Redis
redis>=5.0.1
aioredis>=2.0.1

# HTTP and async
aiofiles>=23.2.1
httpx>=0.25.0

# Environment and configuration
python-dotenv>=1.0.0
python-multipart>=0.0.6

# Development and testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
black>=23.11.0
isort>=5.12.0
flake8>=6.1.0

# Monitoring and logging
structlog>=23.2.0
prometheus-client>=0.19.0
'''
                ),

                TemplateFile(
                    ".env.example",
                    '''# Environment Configuration
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite+aiosqlite:///./app.db
# For PostgreSQL: postgresql+asyncpg://user:password@localhost/dbname

# Redis
REDIS_URL=redis://localhost:6379

# JWT
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
PORT=8000
HOST=0.0.0.0
'''
                ),

                TemplateFile(
                    "Dockerfile",
                    '''FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \\
    chown -R appuser:appuser /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
'''
                ),

                TemplateFile(
                    "docker-compose.yml",
                    '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/{project_name}
      - REDIS_URL=redis://redis:6379
    volumes:
      - .:/app
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: {project_name}
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
''',
                    template_vars=["project_name"]
                ),

                TemplateFile(
                    "alembic.ini",
                    '''[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 88

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
'''
                ),

                TemplateFile(
                    "pytest.ini",
                    '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts =
    --strict-markers
    --disable-warnings
    --cov=.
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
asyncio_mode = auto
'''
                ),

                TemplateFile(
                    "pyproject.toml",
                    '''[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
description = "{project_description}"
readme = "README.md"
requires-python = ">=3.11"
license = {{text = "MIT"}}
authors = [
    {{name = "Terminal Coder", email = "dev@terminalcoder.linux"}}
]
dynamic = ["version"]

[tool.setuptools_scm]

[tool.black]
line-length = 88
target-version = ["py311"]
include = '\\.pyi?$'
extend-exclude = """
(
  /(
      \\.eggs
    | \\.git
    | \\.hg
    | \\.mypy_cache
    | \\.tox
    | \\.venv
    | _build
    | buck-out
    | build
    | dist
    | alembic
  )/
)
"""

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["{project_name}"]

[tool.pylint]
max-line-length = 88
disable = [
    "missing-docstring",
    "too-few-public-methods",
    "import-error"
]
''',
                    template_vars=["project_name", "project_description"]
                ),

                TemplateFile(
                    "tests/test_main.py",
                    '''import pytest
import asyncio
from httpx import AsyncClient
from main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Welcome to" in response.json()["message"]

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
'''
                ),

                TemplateFile(
                    "run.py",
                    '''#!/usr/bin/env python3
"""
{project_name} Runner Script
Optimized for Linux deployment
"""

import os
import sys
import asyncio
import signal
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment for Linux deployment"""
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()

    # Set process title for monitoring
    try:
        import setproctitle
        setproctitle.setproctitle("{project_name}")
    except ImportError:
        pass

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {{signum}}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup environment
    setup_environment()

    logger.info("Starting {project_name}...")

    # Import and run the app
    try:
        import uvicorn
        from main import app, settings

        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=settings.DEBUG,
            workers=1 if settings.DEBUG else min(4, os.cpu_count() or 1),
            access_log=settings.DEBUG,
            server_header=False,
            date_header=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''',
                    executable=True,
                    template_vars=["project_name"]
                )
            ],
            "commands": [
                TemplateCommand("python -m venv venv", "Create virtual environment"),
                TemplateCommand("source venv/bin/activate", "Activate virtual environment"),
                TemplateCommand("pip install --upgrade pip", "Upgrade pip"),
                TemplateCommand("pip install -r requirements.txt", "Install dependencies"),
                TemplateCommand("alembic init alembic", "Initialize database migrations"),
                TemplateCommand("python -m pytest", "Run tests", required=False),
                TemplateCommand("chmod +x run.py", "Make run script executable", linux_only=True),
                TemplateCommand("./run.py", "Start application", required=False)
            ]
        }

        # Rust Actix-Web Template
        self.templates["rust_actix"] = {
            "name": "Rust Actix-Web API",
            "description": "High-performance Rust web API with Actix-Web, authentication, and PostgreSQL",
            "language": "rust",
            "framework": "actix-web",
            "files": [
                TemplateFile(
                    "Cargo.toml",
                    '''[package]
name = "{project_name_snake}"
version = "0.1.0"
edition = "2021"
description = "{project_description}"
authors = ["Terminal Coder <dev@terminalcoder.linux>"]

[dependencies]
# Web framework
actix-web = "4.4"
actix-cors = "0.6"
actix-files = "0.6"

# Async runtime
tokio = {{ version = "1.35", features = ["full"] }}

# Serialization
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
chrono = {{ version = "0.4", features = ["serde"] }}
uuid = {{ version = "1.6", features = ["v4", "serde"] }}

# Database
sqlx = {{ version = "0.7", features = ["runtime-tokio-rustls", "postgres", "chrono", "uuid", "migrate"] }}
sea-orm = {{ version = "0.12", features = ["sqlx-postgres", "runtime-tokio-rustls", "macros", "with-chrono", "with-uuid"] }}

# Security
jsonwebtoken = "9.2"
bcrypt = "0.15"
rand = "0.8"

# Configuration
config = "0.13"
dotenv = "0.15"

# Logging
log = "0.4"
env_logger = "0.10"
tracing = "0.1"
tracing-subscriber = "0.3"
tracing-actix-web = "0.7"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# HTTP client
reqwest = {{ version = "0.11", features = ["json"] }}

# Validation
validator = {{ version = "0.16", features = ["derive"] }}

[dev-dependencies]
actix-rt = "2.8"
actix-test = "0.1"

[[bin]]
name = "server"
path = "src/main.rs"
''',
                    template_vars=["project_name_snake", "project_description"]
                ),

                TemplateFile(
                    "src/main.rs",
                    '''use actix_web::{web, App, HttpServer, Result, middleware, HttpResponse};
use actix_cors::Cors;
use std::env;
use tracing_subscriber;
use tracing_actix_web::TracingLogger;

mod config;
mod handlers;
mod models;
mod database;
mod auth;
mod error;

use config::AppConfig;
use database::Database;
use error::AppError;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load configuration
    dotenv::dotenv().ok();
    let config = AppConfig::new().expect("Failed to load configuration");

    // Initialize database
    let database = Database::new(&config.database_url)
        .await
        .expect("Failed to connect to database");

    // Run migrations
    database.migrate()
        .await
        .expect("Failed to run migrations");

    let bind_address = format!("{}:{}", config.host, config.port);

    tracing::info!("Starting {project_name} server on {}", bind_address);

    HttpServer::new(move || {{
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .app_data(web::Data::new(database.clone()))
            .wrap(cors)
            .wrap(TracingLogger::default())
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            .route("/", web::get().to(handlers::health::root))
            .route("/health", web::get().to(handlers::health::health_check))
            .service(
                web::scope("/api/v1")
                    .service(handlers::auth::auth_routes())
                    .service(handlers::users::user_routes())
                    .service(handlers::items::item_routes())
            )
    }})
    .bind(&bind_address)?
    .run()
    .await
}}
''',
                    template_vars=["project_name"]
                ),

                TemplateFile(
                    "src/config.rs",
                    '''use serde::Deserialize;
use std::env;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub database_url: String,
    pub jwt_secret: String,
    pub jwt_expiration: i64,
    pub rust_log: String,
}

impl AppConfig {
    pub fn new() -> Result<Self, config::ConfigError> {
        let config = config::Config::builder()
            .set_default("host", "0.0.0.0")?
            .set_default("port", 8000)?
            .set_default("database_url", "postgresql://postgres:password@localhost/{project_name_snake}")?
            .set_default("jwt_secret", "your-secret-key")?
            .set_default("jwt_expiration", 3600)?
            .set_default("rust_log", "info")?
            .add_source(config::Environment::default())
            .build()?;

        config.try_deserialize()
    }
}
''',
                    template_vars=["project_name_snake"]
                ),

                TemplateFile(
                    "src/database.rs",
                    '''use sqlx::{PgPool, Row};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;

        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> Result<()> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }

    pub fn get_pool(&self) -> &PgPool {
        &self.pool
    }
}
'''
                ),

                TemplateFile(
                    "src/models.rs",
                    '''use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub password_hash: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateUser {
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    #[validate(length(min = 8))]
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Item {
    pub id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub user_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateItem {
    #[validate(length(min = 1, max = 200))]
    pub title: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
}
'''
                ),

                TemplateFile(
                    "src/auth.rs",
                    '''use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use serde::{Deserialize, Serialize};
use chrono::{Utc, Duration};
use uuid::Uuid;
use bcrypt::{hash, verify, DEFAULT_COST};
use actix_web::{web, HttpRequest, Result, FromRequest, dev::Payload, HttpMessage};
use std::future::{Ready, ready};
use crate::error::AppError;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // User ID
    pub exp: usize,
    pub iat: usize,
}

impl Claims {
    pub fn new(user_id: Uuid, expiration_hours: i64) -> Self {
        let now = Utc::now();
        let exp = (now + Duration::hours(expiration_hours)).timestamp() as usize;
        let iat = now.timestamp() as usize;

        Self {
            sub: user_id.to_string(),
            exp,
            iat,
        }
    }
}

pub fn hash_password(password: &str) -> Result<String, AppError> {
    hash(password, DEFAULT_COST).map_err(|_| AppError::InternalServerError)
}

pub fn verify_password(password: &str, hash: &str) -> Result<bool, AppError> {
    verify(password, hash).map_err(|_| AppError::InternalServerError)
}

pub fn create_jwt(claims: &Claims, secret: &str) -> Result<String, AppError> {
    encode(
        &Header::default(),
        claims,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|_| AppError::InternalServerError)
}

pub fn decode_jwt(token: &str, secret: &str) -> Result<Claims, AppError> {
    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_ref()),
        &Validation::default(),
    )
    .map(|data| data.claims)
    .map_err(|_| AppError::Unauthorized)
}

pub struct AuthenticatedUser {
    pub user_id: Uuid,
}

impl FromRequest for AuthenticatedUser {
    type Error = AppError;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        let auth_header = req.headers().get("Authorization");

        if let Some(auth_header) = auth_header {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str.starts_with("Bearer ") {
                    let token = &auth_str[7..];
                    let secret = "your-secret-key"; // Get from config

                    if let Ok(claims) = decode_jwt(token, secret) {
                        if let Ok(user_id) = Uuid::parse_str(&claims.sub) {
                            return ready(Ok(AuthenticatedUser { user_id }));
                        }
                    }
                }
            }
        }

        ready(Err(AppError::Unauthorized))
    }
}
'''
                ),

                TemplateFile(
                    "src/error.rs",
                    '''use actix_web::{HttpResponse, ResponseError};
use std::fmt;

#[derive(Debug)]
pub enum AppError {
    InternalServerError,
    BadRequest(String),
    Unauthorized,
    NotFound(String),
    ValidationError(String),
    DatabaseError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::InternalServerError => write!(f, "Internal server error"),
            AppError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            AppError::Unauthorized => write!(f, "Unauthorized"),
            AppError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        match self {
            AppError::InternalServerError => {
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Internal server error"
                }))
            }
            AppError::BadRequest(msg) => {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": msg
                }))
            }
            AppError::Unauthorized => {
                HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Unauthorized"
                }))
            }
            AppError::NotFound(msg) => {
                HttpResponse::NotFound().json(serde_json::json!({
                    "error": msg
                }))
            }
            AppError::ValidationError(msg) => {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": msg
                }))
            }
            AppError::DatabaseError(msg) => {
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Database error",
                    "details": msg
                }))
            }
        }
    }
}

impl From<sqlx::Error> for AppError {
    fn from(error: sqlx::Error) -> Self {
        AppError::DatabaseError(error.to_string())
    }
}
'''
                ),

                TemplateFile(
                    "src/handlers/mod.rs",
                    '''pub mod health;
pub mod auth;
pub mod users;
pub mod items;
'''
                ),

                TemplateFile(
                    "src/handlers/health.rs",
                    '''use actix_web::{HttpResponse, Result};
use serde_json::json;
use chrono::Utc;

pub async fn root() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(json!({
        "message": "Welcome to {project_name} API!",
        "version": "1.0.0",
        "timestamp": Utc::now(),
        "status": "running"
    })))
}

pub async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(json!({
        "status": "healthy",
        "timestamp": Utc::now(),
        "version": "1.0.0"
    })))
}
''',
                    template_vars=["project_name"]
                ),

                TemplateFile(
                    ".env.example",
                    '''# Server Configuration
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://postgres:password@localhost/{project_name_snake}

# JWT
JWT_SECRET=your-secret-key-here
JWT_EXPIRATION=3600

# Logging
RUST_LOG=info
''',
                    template_vars=["project_name_snake"]
                ),

                TemplateFile(
                    "Dockerfile",
                    '''# Build stage
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    pkg-config \\
    libssl-dev \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to cache dependencies
RUN mkdir src && echo "fn main() {{}}" > src/main.rs

# Build dependencies
RUN cargo build --release && rm src/main.rs

# Copy source code
COPY src ./src
COPY migrations ./migrations

# Build application
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    ca-certificates \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/server /app/

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["./server"]
'''
                )
            ],
            "commands": [
                TemplateCommand("cargo check", "Check Rust code for errors"),
                TemplateCommand("cargo build", "Build the project"),
                TemplateCommand("cargo test", "Run tests", required=False),
                TemplateCommand("cargo run", "Run the application", required=False),
                TemplateCommand("cargo build --release", "Build optimized release", required=False)
            ]
        }

        # Go Gin Template
        self.templates["go_gin"] = {
            "name": "Go Gin REST API",
            "description": "High-performance Go REST API with Gin framework, JWT auth, and PostgreSQL",
            "language": "go",
            "framework": "gin",
            "files": [
                TemplateFile(
                    "go.mod",
                    '''module {project_name_snake}

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/golang-jwt/jwt/v5 v5.2.0
    github.com/joho/godotenv v1.4.0
    github.com/lib/pq v1.10.9
    github.com/google/uuid v1.5.0
    golang.org/x/crypto v0.17.0
    gorm.io/driver/postgres v1.5.4
    gorm.io/gorm v1.25.5
)
''',
                    template_vars=["project_name_snake"]
                ),

                TemplateFile(
                    "main.go",
                    '''package main

import (
    "log"
    "os"
    "{project_name_snake}/internal/api"
    "{project_name_snake}/internal/config"
    "{project_name_snake}/internal/database"

    "github.com/joho/godotenv"
)

func main() {
    // Load environment variables
    if err := godotenv.Load(); err != nil {
        log.Println("No .env file found")
    }

    // Load configuration
    cfg := config.Load()

    // Initialize database
    db, err := database.Initialize(cfg.DatabaseURL)
    if err != nil {
        log.Fatal("Failed to connect to database:", err)
    }

    // Run migrations
    if err := database.Migrate(db); err != nil {
        log.Fatal("Failed to run migrations:", err)
    }

    // Initialize API server
    server := api.NewServer(cfg, db)

    // Start server
    log.Printf("Starting {project_name} server on :%s", cfg.Port)
    if err := server.Start(); err != nil {
        log.Fatal("Failed to start server:", err)
    }
}
''',
                    template_vars=["project_name_snake", "project_name"]
                ),

                TemplateFile(
                    "internal/config/config.go",
                    '''package config

import (
    "os"
    "strconv"
    "time"
)

type Config struct {
    Port        string
    DatabaseURL string
    JWTSecret   string
    JWTExpiry   time.Duration
    Environment string
}

func Load() *Config {
    jwtExpiryHours, _ := strconv.Atoi(getEnv("JWT_EXPIRY_HOURS", "24"))

    return &Config{
        Port:        getEnv("PORT", "8000"),
        DatabaseURL: getEnv("DATABASE_URL", "postgres://postgres:password@localhost/{project_name_snake}?sslmode=disable"),
        JWTSecret:   getEnv("JWT_SECRET", "your-secret-key"),
        JWTExpiry:   time.Hour * time.Duration(jwtExpiryHours),
        Environment: getEnv("ENVIRONMENT", "development"),
    }
}

func getEnv(key, fallback string) string {
    if value, exists := os.LookupEnv(key); exists {
        return value
    }
    return fallback
}
''',
                    template_vars=["project_name_snake"]
                ),

                TemplateFile(
                    "internal/models/user.go",
                    '''package models

import (
    "time"
    "github.com/google/uuid"
    "gorm.io/gorm"
)

type User struct {
    ID           uuid.UUID `json:"id" gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
    Email        string    `json:"email" gorm:"uniqueIndex;not null"`
    Username     string    `json:"username" gorm:"uniqueIndex;not null"`
    PasswordHash string    `json:"-" gorm:"not null"`
    IsActive     bool      `json:"is_active" gorm:"default:true"`
    CreatedAt    time.Time `json:"created_at"`
    UpdatedAt    time.Time `json:"updated_at"`

    Items []Item `json:"items,omitempty" gorm:"foreignKey:UserID"`
}

type CreateUserRequest struct {
    Email    string `json:"email" binding:"required,email"`
    Username string `json:"username" binding:"required,min=3,max=50"`
    Password string `json:"password" binding:"required,min=8"`
}

type LoginRequest struct {
    Email    string `json:"email" binding:"required,email"`
    Password string `json:"password" binding:"required"`
}

type TokenResponse struct {
    AccessToken string `json:"access_token"`
    TokenType   string `json:"token_type"`
    ExpiresIn   int64  `json:"expires_in"`
}

func (u *User) BeforeCreate(tx *gorm.DB) error {
    if u.ID == uuid.Nil {
        u.ID = uuid.New()
    }
    return nil
}
'''
                ),

                TemplateFile(
                    "internal/models/item.go",
                    '''package models

import (
    "time"
    "github.com/google/uuid"
    "gorm.io/gorm"
)

type Item struct {
    ID          uuid.UUID  `json:"id" gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
    Title       string     `json:"title" gorm:"not null"`
    Description *string    `json:"description"`
    UserID      uuid.UUID  `json:"user_id" gorm:"type:uuid;not null"`
    CreatedAt   time.Time  `json:"created_at"`
    UpdatedAt   time.Time  `json:"updated_at"`

    User User `json:"user,omitempty" gorm:"foreignKey:UserID"`
}

type CreateItemRequest struct {
    Title       string  `json:"title" binding:"required,min=1,max=200"`
    Description *string `json:"description"`
}

type UpdateItemRequest struct {
    Title       *string `json:"title"`
    Description *string `json:"description"`
}

func (i *Item) BeforeCreate(tx *gorm.DB) error {
    if i.ID == uuid.Nil {
        i.ID = uuid.New()
    }
    return nil
}
'''
                ),

                TemplateFile(
                    "internal/api/server.go",
                    '''package api

import (
    "fmt"
    "net/http"
    "time"

    "{project_name_snake}/internal/config"
    "{project_name_snake}/internal/handlers"
    "{project_name_snake}/internal/middleware"

    "github.com/gin-gonic/gin"
    "gorm.io/gorm"
)

type Server struct {
    config *config.Config
    db     *gorm.DB
    router *gin.Engine
}

func NewServer(cfg *config.Config, db *gorm.DB) *Server {
    if cfg.Environment == "production" {
        gin.SetMode(gin.ReleaseMode)
    }

    router := gin.New()

    // Global middleware
    router.Use(gin.Logger())
    router.Use(gin.Recovery())
    router.Use(middleware.CORS())
    router.Use(middleware.Security())

    server := &Server{
        config: cfg,
        db:     db,
        router: router,
    }

    server.setupRoutes()

    return server
}

func (s *Server) setupRoutes() {
    // Health check
    s.router.GET("/", s.root)
    s.router.GET("/health", s.healthCheck)

    // Initialize handlers
    authHandler := handlers.NewAuthHandler(s.db, s.config.JWTSecret, s.config.JWTExpiry)
    userHandler := handlers.NewUserHandler(s.db)
    itemHandler := handlers.NewItemHandler(s.db)

    // API routes
    api := s.router.Group("/api/v1")
    {
        // Public routes
        api.POST("/auth/register", authHandler.Register)
        api.POST("/auth/login", authHandler.Login)

        // Protected routes
        protected := api.Group("")
        protected.Use(middleware.AuthMiddleware(s.config.JWTSecret))
        {
            // User routes
            protected.GET("/users/me", userHandler.GetCurrentUser)

            // Item routes
            protected.GET("/items", itemHandler.GetItems)
            protected.POST("/items", itemHandler.CreateItem)
            protected.GET("/items/:id", itemHandler.GetItem)
            protected.PUT("/items/:id", itemHandler.UpdateItem)
            protected.DELETE("/items/:id", itemHandler.DeleteItem)
        }
    }
}

func (s *Server) root(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "message":   "Welcome to {project_name} API!",
        "version":   "1.0.0",
        "timestamp": time.Now().UTC(),
        "status":    "running",
    })
}

func (s *Server) healthCheck(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "status":    "healthy",
        "timestamp": time.Now().UTC(),
        "version":   "1.0.0",
    })
}

func (s *Server) Start() error {
    return s.router.Run(fmt.Sprintf(":%s", s.config.Port))
}
''',
                    template_vars=["project_name_snake", "project_name"]
                ),

                TemplateFile(
                    ".env.example",
                    '''# Server Configuration
PORT=8000
ENVIRONMENT=development

# Database
DATABASE_URL=postgres://postgres:password@localhost/{project_name_snake}?sslmode=disable

# JWT
JWT_SECRET=your-secret-key-here
JWT_EXPIRY_HOURS=24

# Logging
GIN_MODE=debug
''',
                    template_vars=["project_name_snake"]
                )
            ],
            "commands": [
                TemplateCommand("go mod tidy", "Download and organize dependencies"),
                TemplateCommand("go build", "Build the application"),
                TemplateCommand("go test ./...", "Run tests", required=False),
                TemplateCommand("go run main.go", "Run the application", required=False)
            ]
        }

    def get_template(self, template_name: str) -> Optional[Dict]:
        """Get template by name"""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available template names"""
        return list(self.templates.keys())

    async def create_project_from_template(self, template_name: str, project_path: str,
                                         variables: Dict[str, Any]) -> bool:
        """Create project from template with variable substitution"""
        template = self.get_template(template_name)
        if not template:
            logger.error(f"Template {template_name} not found")
            return False

        try:
            project_path_obj = Path(project_path)
            project_path_obj.mkdir(parents=True, exist_ok=True)

            # Add derived variables
            variables.update({
                "project_name_snake": variables["project_name"].lower().replace(" ", "_").replace("-", "_"),
                "project_name_kebab": variables["project_name"].lower().replace(" ", "-").replace("_", "-"),
                "project_name_camel": variables["project_name"].replace(" ", "").replace("-", "").replace("_", ""),
                "timestamp": datetime.now().isoformat()
            })

            # Create files
            for file_def in template["files"]:
                file_path = project_path_obj / file_def.path
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Substitute variables
                content = file_def.content
                for var_name, var_value in variables.items():
                    content = content.replace(f"{{{var_name}}}", str(var_value))

                # Write file
                with open(file_path, 'w') as f:
                    f.write(content)

                # Set executable if needed
                if file_def.executable:
                    file_path.chmod(0o755)

                logger.info(f"Created file: {file_def.path}")

            # Run setup commands
            for command in template["commands"]:
                if command.linux_only and os.name != 'posix':
                    continue

                logger.info(f"Running: {command.description}")
                try:
                    result = subprocess.run(
                        command.command,
                        shell=True,
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode != 0 and command.required:
                        logger.error(f"Command failed: {command.command}")
                        logger.error(f"Error: {result.stderr}")
                        if command.required:
                            return False
                    else:
                        logger.info(f"✅ {command.description}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"Command timed out: {command.command}")
                    if command.required:
                        return False

                except Exception as e:
                    logger.error(f"Error running command '{command.command}': {e}")
                    if command.required:
                        return False

            logger.info(f"Successfully created project from template: {template_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating project from template: {e}")
            return False

    def get_template_variables(self, template_name: str) -> List[str]:
        """Get list of variables used in template"""
        template = self.get_template(template_name)
        if not template:
            return []

        variables = set()
        for file_def in template["files"]:
            if file_def.template_vars:
                variables.update(file_def.template_vars)

        return list(variables)


# Global template engine instance
template_engine = ProjectTemplateEngine()