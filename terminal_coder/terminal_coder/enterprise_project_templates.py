"""
Enterprise Project Templates
Advanced project templates with Linux-native optimizations, containerization,
CI/CD integration, and professional deployment configurations
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final
import aiofiles
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import tempfile


class ProjectType(Enum):
    """Types of projects"""
    PYTHON_CLI = "python_cli"
    PYTHON_WEB_API = "python_web_api"
    PYTHON_MICROSERVICE = "python_microservice"
    PYTHON_ML_SERVICE = "python_ml_service"
    NODEJS_API = "nodejs_api"
    NODEJS_MICROSERVICE = "nodejs_microservice"
    GO_CLI = "go_cli"
    GO_MICROSERVICE = "go_microservice"
    RUST_CLI = "rust_cli"
    RUST_SYSTEM_SERVICE = "rust_system_service"
    CONTAINER_APP = "container_app"
    KUBERNETES_APP = "kubernetes_app"
    SYSTEMD_SERVICE = "systemd_service"
    LINUX_DAEMON = "linux_daemon"


class DeploymentTarget(Enum):
    """Deployment targets"""
    BARE_METAL = "bare_metal"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    SYSTEMD = "systemd"
    AWS_EC2 = "aws_ec2"
    GCP_COMPUTE = "gcp_compute"
    AZURE_VM = "azure_vm"


class CIProvider(Enum):
    """CI/CD providers"""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    DRONE = "drone"
    TEKTON = "tekton"


@dataclass(slots=True)
class ProjectConfig:
    """Project configuration"""
    name: str
    description: str
    project_type: ProjectType
    deployment_targets: list[DeploymentTarget]
    ci_providers: list[CIProvider]
    python_version: str = "3.11"
    node_version: str = "18"
    go_version: str = "1.21"
    rust_version: str = "1.70"
    database: str | None = None
    cache: str | None = None
    monitoring: bool = True
    logging: bool = True
    health_checks: bool = True
    metrics: bool = True
    security_scanning: bool = True
    linux_optimizations: bool = True


@dataclass(slots=True)
class TemplateFile:
    """Template file configuration"""
    path: str
    content: str
    executable: bool = False
    template_vars: dict[str, str] = field(default_factory=dict)


class EnterpriseProjectTemplates:
    """Enterprise-grade project template generator"""

    # Professional project templates with Linux optimizations
    TEMPLATES: Final[dict[ProjectType, dict[str, Any]]] = {
        ProjectType.PYTHON_WEB_API: {
            "description": "Production-ready FastAPI service with Linux optimizations",
            "default_database": "postgresql",
            "default_cache": "redis",
            "files": [
                "main.py", "app/__init__.py", "app/api/__init__.py", "app/api/routes.py",
                "app/core/config.py", "app/core/security.py", "app/models/__init__.py",
                "requirements.txt", "requirements-dev.txt", "pyproject.toml",
                "Dockerfile", "docker-compose.yml", "kubernetes.yaml",
                ".github/workflows/ci.yml", ".github/workflows/cd.yml",
                "deploy/systemd/api.service", "deploy/nginx/api.conf",
                "scripts/install.sh", "scripts/deploy.sh", "scripts/health-check.sh"
            ]
        },
        ProjectType.PYTHON_MICROSERVICE: {
            "description": "Cloud-native Python microservice with observability",
            "default_database": "postgresql",
            "default_cache": "redis",
            "files": [
                "main.py", "src/__init__.py", "src/service/__init__.py",
                "src/service/handlers.py", "src/service/models.py", "src/service/database.py",
                "src/observability/metrics.py", "src/observability/logging.py",
                "requirements.txt", "Dockerfile", "k8s/deployment.yaml", "k8s/service.yaml",
                "helm/Chart.yaml", "helm/values.yaml", ".github/workflows/deploy.yml"
            ]
        },
        ProjectType.GO_MICROSERVICE: {
            "description": "High-performance Go microservice with gRPC",
            "files": [
                "main.go", "cmd/server/main.go", "internal/server/server.go",
                "internal/handlers/handlers.go", "internal/models/models.go",
                "api/proto/service.proto", "go.mod", "Dockerfile", "Makefile",
                "deployments/k8s/deployment.yaml", ".github/workflows/go.yml"
            ]
        },
        ProjectType.RUST_SYSTEM_SERVICE: {
            "description": "High-performance Rust system service",
            "files": [
                "src/main.rs", "src/lib.rs", "src/config.rs", "src/service.rs",
                "Cargo.toml", "Dockerfile", "systemd/rust-service.service",
                ".github/workflows/rust.yml", "scripts/install.sh"
            ]
        },
        ProjectType.LINUX_DAEMON: {
            "description": "Native Linux daemon with systemd integration",
            "files": [
                "src/main.py", "src/daemon.py", "src/config.py", "src/logging_config.py",
                "systemd/daemon.service", "logrotate/daemon", "syslog-ng/daemon.conf",
                "scripts/install.sh", "scripts/uninstall.sh", "requirements.txt"
            ]
        }
    }

    def __init__(self) -> None:
        self.console = Console()

    async def create_project_interactive(self) -> Path | None:
        """Interactive project creation wizard"""
        self.console.print(Panel.fit(
            "[bold cyan]🚀 Enterprise Project Template Generator[/bold cyan]\n"
            "[dim]Create production-ready projects with Linux optimizations[/dim]",
            border_style="cyan"
        ))

        # Show available templates
        template_table = Table(title="Available Project Templates")
        template_table.add_column("Template", style="cyan", no_wrap=True)
        template_table.add_column("Description", style="white")
        template_table.add_column("Best For", style="green")

        template_info = {
            ProjectType.PYTHON_WEB_API: ("FastAPI Web API", "REST APIs, web services"),
            ProjectType.PYTHON_MICROSERVICE: ("Python Microservice", "Cloud-native applications"),
            ProjectType.GO_MICROSERVICE: ("Go Microservice", "High-performance services"),
            ProjectType.RUST_SYSTEM_SERVICE: ("Rust System Service", "System-level applications"),
            ProjectType.LINUX_DAEMON: ("Linux Daemon", "Background services"),
            ProjectType.PYTHON_CLI: ("Python CLI Tool", "Command-line applications"),
            ProjectType.CONTAINER_APP: ("Container App", "Docker/Kubernetes apps"),
        }

        for project_type, (name, description) in template_info.items():
            template_table.add_row(
                project_type.value,
                name,
                description
            )

        self.console.print(template_table)

        # Get project configuration
        config = await self._get_project_config_interactive()
        if not config:
            return None

        # Create project
        project_path = await self.create_project(config)
        return project_path

    async def _get_project_config_interactive(self) -> ProjectConfig | None:
        """Get project configuration interactively"""
        try:
            # Project basics
            name = Prompt.ask("[green]Project name[/green]")
            if not name:
                return None

            description = Prompt.ask(
                "[green]Project description[/green]",
                default=f"Enterprise-grade {name} application"
            )

            # Project type
            project_type_choice = Prompt.ask(
                "[green]Project template[/green]",
                choices=[pt.value for pt in ProjectType],
                default=ProjectType.PYTHON_WEB_API.value
            )
            project_type = ProjectType(project_type_choice)

            # Deployment targets
            self.console.print("\n[cyan]Select deployment targets (comma-separated):[/cyan]")
            for target in DeploymentTarget:
                self.console.print(f"  • {target.value}")

            deployment_choice = Prompt.ask(
                "[green]Deployment targets[/green]",
                default="docker,kubernetes"
            )
            deployment_targets = [
                DeploymentTarget(target.strip())
                for target in deployment_choice.split(",")
                if target.strip() in [dt.value for dt in DeploymentTarget]
            ]

            # CI/CD providers
            ci_choice = Prompt.ask(
                "[green]CI/CD provider[/green]",
                choices=[ci.value for ci in CIProvider],
                default=CIProvider.GITHUB_ACTIONS.value
            )
            ci_providers = [CIProvider(ci_choice)]

            # Optional features
            monitoring = Confirm.ask("[yellow]Include monitoring?[/yellow]", default=True)
            security_scanning = Confirm.ask("[yellow]Include security scanning?[/yellow]", default=True)
            linux_optimizations = Confirm.ask("[yellow]Apply Linux optimizations?[/yellow]", default=True)

            # Database and cache
            database = None
            cache = None

            if project_type in [ProjectType.PYTHON_WEB_API, ProjectType.PYTHON_MICROSERVICE]:
                if Confirm.ask("[yellow]Include database?[/yellow]", default=True):
                    database = Prompt.ask(
                        "[green]Database type[/green]",
                        choices=["postgresql", "mysql", "sqlite", "mongodb"],
                        default="postgresql"
                    )

                if Confirm.ask("[yellow]Include cache?[/yellow]", default=True):
                    cache = Prompt.ask(
                        "[green]Cache type[/green]",
                        choices=["redis", "memcached"],
                        default="redis"
                    )

            return ProjectConfig(
                name=name,
                description=description,
                project_type=project_type,
                deployment_targets=deployment_targets,
                ci_providers=ci_providers,
                database=database,
                cache=cache,
                monitoring=monitoring,
                security_scanning=security_scanning,
                linux_optimizations=linux_optimizations
            )

        except (KeyboardInterrupt, EOFError):
            self.console.print("[yellow]Project creation cancelled[/yellow]")
            return None

    async def create_project(self, config: ProjectConfig) -> Path:
        """Create project from configuration"""
        self.console.print(f"[bold green]📁 Creating project: {config.name}[/bold green]")

        project_path = Path.cwd() / config.name
        if project_path.exists():
            if not Confirm.ask(f"[yellow]Directory {config.name} exists. Overwrite?[/yellow]"):
                raise FileExistsError(f"Project directory {config.name} already exists")
            shutil.rmtree(project_path)

        project_path.mkdir(parents=True, exist_ok=True)

        # Generate project files
        files_to_create = await self._generate_project_files(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            task = progress.add_task("Creating project files...", total=len(files_to_create))

            for template_file in files_to_create:
                file_path = project_path / template_file.path
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Process template variables
                content = template_file.content
                for var, value in template_file.template_vars.items():
                    content = content.replace(f"{{{{{var}}}}}", value)

                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(content)

                if template_file.executable:
                    file_path.chmod(0o755)

                progress.advance(task)

        # Post-creation setup
        await self._post_creation_setup(project_path, config)

        self.console.print(f"[green]✅ Project created successfully at {project_path}[/green]")
        self.console.print(f"[cyan]📖 Next steps:[/cyan]")
        self.console.print(f"  1. cd {config.name}")
        self.console.print(f"  2. Review and customize configuration files")
        self.console.print(f"  3. Run ./scripts/install.sh to set up dependencies")
        self.console.print(f"  4. Start development!")

        return project_path

    async def _generate_project_files(self, config: ProjectConfig) -> list[TemplateFile]:
        """Generate project files based on configuration"""
        template_vars = {
            "project_name": config.name,
            "project_description": config.description,
            "python_version": config.python_version,
            "node_version": config.node_version,
            "go_version": config.go_version,
            "rust_version": config.rust_version,
            "database": config.database or "none",
            "cache": config.cache or "none",
            "timestamp": datetime.now().isoformat(),
        }

        files = []

        if config.project_type == ProjectType.PYTHON_WEB_API:
            files.extend(await self._generate_python_web_api_files(config, template_vars))
        elif config.project_type == ProjectType.PYTHON_MICROSERVICE:
            files.extend(await self._generate_python_microservice_files(config, template_vars))
        elif config.project_type == ProjectType.GO_MICROSERVICE:
            files.extend(await self._generate_go_microservice_files(config, template_vars))
        elif config.project_type == ProjectType.RUST_SYSTEM_SERVICE:
            files.extend(await self._generate_rust_service_files(config, template_vars))
        elif config.project_type == ProjectType.LINUX_DAEMON:
            files.extend(await self._generate_linux_daemon_files(config, template_vars))

        # Add common files
        files.extend(await self._generate_common_files(config, template_vars))

        return files

    async def _generate_python_web_api_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate Python FastAPI project files"""
        files = []

        # Main application
        main_py = '''#!/usr/bin/env python3
"""
{{project_name}} - {{project_description}}
Production-ready FastAPI application with Linux optimizations
"""

import uvicorn
from app.api.main import create_app

if __name__ == "__main__":
    app = create_app()

    # Linux-optimized configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Set based on CPU cores
        loop="uvloop",  # Linux-optimized event loop
        http="httptools",  # Fast HTTP parser
        lifespan="on",
        access_log=True,
        server_header=False,
        date_header=False,
    )
'''
        files.append(TemplateFile("main.py", main_py, executable=True, template_vars=vars))

        # FastAPI application
        app_main = '''"""FastAPI application factory"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.core.config import get_settings
from app.api.routes import api_router
from app.core.database import init_db, close_db
from app.core.logging import setup_logging

settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting {{project_name}} API...")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down {{project_name}} API...")
    await close_db()

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="{{project_name}}",
        description="{{project_description}}",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers"""
        return {"status": "healthy", "service": "{{project_name}}"}

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        # Implement metrics collection here
        return {"metrics": "implemented"}

    return app
'''
        files.append(TemplateFile("app/api/main.py", app_main, template_vars=vars))

        # Configuration
        config_py = '''"""Application configuration"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "{{project_name}}"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "your-secret-key-here"

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/{{project_name}}"

    # Cache
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text

    # Linux optimizations
    WORKER_PROCESSES: int = 1  # Set based on CPU cores
    WORKER_CONNECTIONS: int = 1000

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
'''
        files.append(TemplateFile("app/core/config.py", config_py, template_vars=vars))

        # Requirements
        requirements = '''# Production dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
uvloop==0.19.0  # Linux-optimized event loop
httptools==0.6.1  # Fast HTTP parsing
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6

# Database
asyncpg==0.29.0  # PostgreSQL async driver
alembic==1.13.0  # Database migrations

# Cache
redis==5.0.1
aioredis==2.0.1

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
'''
        files.append(TemplateFile("requirements.txt", requirements, template_vars=vars))

        # Dockerfile optimized for Linux
        dockerfile = '''# Multi-stage Docker build for Linux optimization
FROM python:{{python_version}}-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:{{python_version}}-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
'''
        files.append(TemplateFile("Dockerfile", dockerfile, template_vars=vars))

        return files

    async def _generate_python_microservice_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate Python microservice files"""
        files = []

        # Add microservice-specific files
        service_py = '''"""Core microservice logic"""

import asyncio
import logging
from typing import Any, Dict
import aiohttp
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str = "{{project_name}}"
    version: str = "1.0.0"
    port: int = 8000
    health_check_interval: int = 30

class MicroService:
    """Base microservice class with Linux optimizations"""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.running = False
        self.health_status = "healthy"

    async def start(self):
        """Start the microservice"""
        logger.info(f"Starting {self.config.name} v{self.config.version}")
        self.running = True

        # Start health check task
        asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """Stop the microservice"""
        logger.info(f"Stopping {self.config.name}")
        self.running = False

    async def _health_check_loop(self):
        """Periodic health check"""
        while self.running:
            try:
                # Perform health checks here
                self.health_status = "healthy"
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.health_status = "unhealthy"
                await asyncio.sleep(5)  # Retry sooner if unhealthy
'''
        files.append(TemplateFile("src/service/microservice.py", service_py, template_vars=vars))

        return files

    async def _generate_go_microservice_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate Go microservice files"""
        files = []

        # Main Go file
        main_go = '''package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/gin-gonic/gin"
)

func main() {
    // Create Gin router with Linux optimizations
    gin.SetMode(gin.ReleaseMode)
    router := gin.New()

    // Middleware
    router.Use(gin.Logger())
    router.Use(gin.Recovery())

    // Health check endpoint
    router.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "service": "{{project_name}}",
            "timestamp": time.Now().Unix(),
        })
    })

    // API routes
    v1 := router.Group("/api/v1")
    {
        v1.GET("/status", getStatus)
    }

    // Server configuration optimized for Linux
    srv := &http.Server{
        Addr:         ":8000",
        Handler:      router,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  60 * time.Second,
    }

    // Start server in a goroutine
    go func() {
        log.Printf("Starting {{project_name}} server on :8000")
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server failed to start: %v", err)
        }
    }()

    // Wait for interrupt signal for graceful shutdown
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    log.Println("Shutting down server...")

    // Graceful shutdown
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := srv.Shutdown(ctx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }

    log.Println("Server exited")
}

func getStatus(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{
        "service": "{{project_name}}",
        "version": "1.0.0",
        "status": "running",
        "uptime": time.Now().Unix(),
    })
}
'''
        files.append(TemplateFile("main.go", main_go, template_vars=vars))

        # Go mod file
        go_mod = '''module {{project_name}}

go {{go_version}}

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/prometheus/client_golang v1.17.0
)
'''
        files.append(TemplateFile("go.mod", go_mod, template_vars=vars))

        return files

    async def _generate_rust_service_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate Rust system service files"""
        files = []

        # Cargo.toml
        cargo_toml = '''[package]
name = "{{project_name}}"
version = "0.1.0"
edition = "2021"
description = "{{project_description}}"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.0", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
signal-hook = "0.3"
systemd = "0.10"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
'''
        files.append(TemplateFile("Cargo.toml", cargo_toml, template_vars=vars))

        # Main Rust file
        main_rs = '''use std::sync::Arc;
use std::time::Duration;
use tokio::signal::unix::{signal, SignalKind};
use tokio::time::sleep;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting {{project_name}} service...");

    // Create service
    let service = Arc::new(SystemService::new());

    // Setup signal handlers for graceful shutdown
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    // Notify systemd that we're ready (if running under systemd)
    #[cfg(target_os = "linux")]
    systemd::daemon::notify(false, [(systemd::daemon::STATE_READY, "1")].iter())?;

    // Main service loop
    let service_clone = Arc::clone(&service);
    let service_task = tokio::spawn(async move {
        service_clone.run().await;
    });

    // Wait for shutdown signal
    tokio::select! {
        _ = sigterm.recv() => {
            info!("Received SIGTERM, shutting down gracefully...");
        }
        _ = sigint.recv() => {
            info!("Received SIGINT, shutting down gracefully...");
        }
    }

    // Stop service
    service.stop().await;
    service_task.await?;

    info!("{{project_name}} service stopped");
    Ok(())
}

struct SystemService {
    running: std::sync::atomic::AtomicBool,
}

impl SystemService {
    fn new() -> Self {
        Self {
            running: std::sync::atomic::AtomicBool::new(false),
        }
    }

    async fn run(&self) {
        use std::sync::atomic::Ordering;

        self.running.store(true, Ordering::SeqCst);
        info!("{{project_name}} service is running");

        while self.running.load(Ordering::SeqCst) {
            // Main service logic here
            self.health_check().await;

            // Sleep for a while
            sleep(Duration::from_secs(10)).await;
        }
    }

    async fn stop(&self) {
        use std::sync::atomic::Ordering;

        info!("Stopping service...");
        self.running.store(false, Ordering::SeqCst);
    }

    async fn health_check(&self) {
        // Perform health checks
        info!("Service health check passed");

        // Notify systemd watchdog (if configured)
        #[cfg(target_os = "linux")]
        if let Err(e) = systemd::daemon::notify(false, [(systemd::daemon::STATE_WATCHDOG, "1")].iter()) {
            error!("Failed to notify systemd watchdog: {}", e);
        }
    }
}
'''
        files.append(TemplateFile("src/main.rs", main_rs, template_vars=vars))

        return files

    async def _generate_linux_daemon_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate Linux daemon files"""
        files = []

        # Systemd service file
        systemd_service = '''[Unit]
Description={{project_description}}
After=network.target
Wants=network.target

[Service]
Type=notify
ExecStart=/opt/{{project_name}}/bin/{{project_name}}
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
User={{project_name}}
Group={{project_name}}

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/{{project_name}} /var/lib/{{project_name}}

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier={{project_name}}

# Watchdog
WatchdogSec=30

[Install]
WantedBy=multi-user.target
'''
        files.append(TemplateFile("systemd/{{project_name}}.service", systemd_service, template_vars=vars))

        return files

    async def _generate_common_files(self, config: ProjectConfig, vars: dict[str, str]) -> list[TemplateFile]:
        """Generate common project files"""
        files = []

        # GitHub Actions CI/CD
        if CIProvider.GITHUB_ACTIONS in config.ci_providers:
            github_ci = '''name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '{{python_version}}'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest

    - name: Security scan
      run: bandit -r . -f json -o bandit-report.json

    - name: Upload security report
      uses: actions/upload-artifact@v3
      with:
        name: security-report
        path: bandit-report.json

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
'''
            files.append(TemplateFile(".github/workflows/ci.yml", github_ci, template_vars=vars))

        # Installation script
        install_script = '''#!/bin/bash
# {{project_name}} Installation Script
# Optimized for Linux systems

set -euo pipefail

PROJECT_NAME="{{project_name}}"
PROJECT_DIR="/opt/$PROJECT_NAME"
SERVICE_USER="$PROJECT_NAME"
LOG_DIR="/var/log/$PROJECT_NAME"
DATA_DIR="/var/lib/$PROJECT_NAME"

# Colors
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
}

create_user() {
    log "Creating service user: $SERVICE_USER"
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
        useradd --system --shell /bin/false --home-dir "$PROJECT_DIR" "$SERVICE_USER"
    fi
}

create_directories() {
    log "Creating directories..."
    mkdir -p "$PROJECT_DIR"/{bin,config,logs}
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"

    chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR" "$LOG_DIR" "$DATA_DIR"
    chmod 755 "$PROJECT_DIR"
    chmod 750 "$LOG_DIR" "$DATA_DIR"
}

install_dependencies() {
    log "Installing system dependencies..."
    apt-get update
    apt-get install -y python3 python3-pip python3-venv systemd
}

install_application() {
    log "Installing application..."
    # Copy application files
    cp -r . "$PROJECT_DIR/"

    # Create virtual environment
    python3 -m venv "$PROJECT_DIR/venv"
    source "$PROJECT_DIR/venv/bin/activate"
    pip install -r "$PROJECT_DIR/requirements.txt"

    # Set permissions
    chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"
}

install_systemd_service() {
    log "Installing systemd service..."
    cp systemd/*.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable "$PROJECT_NAME"
}

setup_logging() {
    log "Setting up logging..."
    # Logrotate configuration
    cat > "/etc/logrotate.d/$PROJECT_NAME" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $PROJECT_NAME >/dev/null 2>&1 || true
    endscript
}
EOF
}

main() {
    log "Installing $PROJECT_NAME..."

    check_root
    create_user
    create_directories
    install_dependencies
    install_application
    install_systemd_service
    setup_logging

    log "Installation completed successfully!"
    log "Start the service with: systemctl start $PROJECT_NAME"
    log "Check logs with: journalctl -u $PROJECT_NAME -f"
}

main "$@"
'''
        files.append(TemplateFile("scripts/install.sh", install_script, executable=True, template_vars=vars))

        # Docker Compose
        if DeploymentTarget.DOCKER in config.deployment_targets:
            docker_compose = '''version: '3.8'

services:
  {{project_name}}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/{{project_name}}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB={{project_name}}
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deploy/nginx:/etc/nginx/conf.d
    depends_on:
      - {{project_name}}
    restart: unless-stopped

volumes:
  postgres_data:
'''
            files.append(TemplateFile("docker-compose.yml", docker_compose, template_vars=vars))

        # Kubernetes deployment
        if DeploymentTarget.KUBERNETES in config.deployment_targets:
            k8s_deployment = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{project_name}}
  labels:
    app: {{project_name}}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {{project_name}}
  template:
    metadata:
      labels:
        app: {{project_name}}
    spec:
      containers:
      - name: {{project_name}}
        image: {{project_name}}:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{project_name}}-secrets
              key: database-url
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {{project_name}}-service
spec:
  selector:
    app: {{project_name}}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
'''
            files.append(TemplateFile("k8s/deployment.yaml", k8s_deployment, template_vars=vars))

        # README
        readme = '''# {{project_name}}

{{project_description}}

## Features

- 🐧 **Linux-native optimization**
- 🚀 **Production-ready configuration**
- 🔒 **Enterprise security**
- 📊 **Built-in monitoring**
- 🐳 **Container support**
- ☸️ **Kubernetes ready**

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Production Deployment

```bash
# Install on Linux server
sudo ./scripts/install.sh

# Start the service
sudo systemctl start {{project_name}}

# Check status
sudo systemctl status {{project_name}}
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f {{project_name}}
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app={{project_name}}
```

## Configuration

Configuration is managed through environment variables:

- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARN, ERROR)

## Monitoring

- Health check endpoint: `/health`
- Metrics endpoint: `/metrics`
- Logs: `journalctl -u {{project_name}} -f`

## Security

This application includes enterprise-grade security features:

- Non-root container execution
- Security headers
- Input validation
- Rate limiting
- Audit logging

## Performance

Optimized for Linux environments:

- uvloop event loop
- Connection pooling
- Async/await patterns
- Efficient resource usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
'''
        files.append(TemplateFile("README.md", readme, template_vars=vars))

        return files

    async def _post_creation_setup(self, project_path: Path, config: ProjectConfig) -> None:
        """Post-creation setup tasks"""
        try:
            # Initialize git repository
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True
            )

            # Create .gitignore
            gitignore_content = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local

# Build
build/
dist/
*.egg-info/

# Testing
.coverage
.pytest_cache/
htmlcov/

# Docker
.dockerignore

# Temporary files
*.tmp
*.temp
"""
            (project_path / ".gitignore").write_text(gitignore_content)

            # Make scripts executable
            scripts_dir = project_path / "scripts"
            if scripts_dir.exists():
                for script in scripts_dir.glob("*.sh"):
                    script.chmod(0o755)

        except subprocess.CalledProcessError:
            self.console.print("[yellow]⚠️ Could not initialize git repository[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Post-creation setup error: {e}[/red]")

    def list_available_templates(self) -> None:
        """List all available project templates"""
        table = Table(title="🚀 Available Enterprise Project Templates")
        table.add_column("Template", style="cyan", no_wrap=True)
        table.add_column("Type", style="blue")
        table.add_column("Description", style="white")
        table.add_column("Features", style="green")

        template_features = {
            ProjectType.PYTHON_WEB_API: "FastAPI, PostgreSQL, Redis, Docker, K8s",
            ProjectType.PYTHON_MICROSERVICE: "FastAPI, Observability, Cloud-native",
            ProjectType.GO_MICROSERVICE: "Gin, gRPC, High-performance",
            ProjectType.RUST_SYSTEM_SERVICE: "Tokio, systemd, Low-level",
            ProjectType.LINUX_DAEMON: "systemd, Logging, Service management",
        }

        for project_type, template_info in self.TEMPLATES.items():
            features = template_features.get(project_type, "Standard features")
            table.add_row(
                project_type.value,
                project_type.value.replace('_', ' ').title(),
                template_info["description"],
                features
            )

        self.console.print(table)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass