"""
Distributed Computing and Cloud Integration
Advanced cloud computing, distributed systems, and containerization support
"""

import asyncio
import aiohttp
import json
import os
import logging
import time
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import tempfile
from datetime import datetime, timedelta
import yaml

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1, container_v1
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CloudResource:
    """Cloud resource representation"""
    provider: str
    resource_type: str
    name: str
    region: str
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost_estimate: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class ContainerSpec:
    """Container specification for deployment"""
    image: str
    name: str
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    memory_limit: str = "512Mi"
    cpu_limit: str = "500m"
    replicas: int = 1
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    success: bool
    deployment_id: str
    endpoint: Optional[str] = None
    status: str = "pending"
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DockerManager:
    """Docker container management"""

    def __init__(self):
        self.client = None
        self.containers = {}
        self.images = {}

    async def initialize(self) -> bool:
        """Initialize Docker client"""
        try:
            if not DOCKER_AVAILABLE:
                logger.warning("Docker library not available")
                return False

            self.client = docker.from_env()

            # Test connection
            self.client.ping()

            logger.info("Docker client initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Docker: {e}")
            return False

    async def build_image(self, dockerfile_path: str, image_name: str,
                         build_args: Optional[Dict[str, str]] = None) -> bool:
        """Build Docker image from Dockerfile"""
        try:
            if not self.client:
                return False

            dockerfile_dir = Path(dockerfile_path).parent

            # Build image
            logger.info(f"Building Docker image: {image_name}")

            build_kwargs = {
                "path": str(dockerfile_dir),
                "tag": image_name,
                "rm": True,
                "forcerm": True
            }

            if build_args:
                build_kwargs["buildargs"] = build_args

            image, build_logs = self.client.images.build(**build_kwargs)

            # Log build output
            for log_line in build_logs:
                if 'stream' in log_line:
                    logger.debug(f"Docker build: {log_line['stream'].strip()}")

            self.images[image_name] = image
            logger.info(f"Successfully built image: {image_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to build Docker image {image_name}: {e}")
            return False

    async def run_container(self, spec: ContainerSpec) -> Optional[str]:
        """Run container from specification"""
        try:
            if not self.client:
                return None

            # Prepare container configuration
            container_config = {
                "image": spec.image,
                "name": spec.name,
                "detach": True,
                "remove": False,
                "mem_limit": spec.memory_limit,
                "cpu_quota": int(float(spec.cpu_limit.rstrip('m')) * 1000),
                "cpu_period": 100000
            }

            if spec.command:
                container_config["command"] = spec.command

            if spec.args:
                container_config["command"] = (container_config.get("command", []) + spec.args)

            if spec.env_vars:
                container_config["environment"] = spec.env_vars

            if spec.ports:
                container_config["ports"] = {f"{port}/tcp": port for port in spec.ports}

            if spec.labels:
                container_config["labels"] = spec.labels

            # Run container
            container = self.client.containers.run(**container_config)

            self.containers[spec.name] = container

            logger.info(f"Started container: {spec.name} (ID: {container.short_id})")
            return container.id

        except Exception as e:
            logger.error(f"Failed to run container {spec.name}: {e}")
            return None

    async def get_container_logs(self, container_name: str, tail: int = 100) -> List[str]:
        """Get container logs"""
        try:
            if container_name in self.containers:
                container = self.containers[container_name]
                logs = container.logs(tail=tail, stream=False).decode('utf-8')
                return logs.split('\n')
            return []
        except Exception as e:
            logger.error(f"Failed to get logs for {container_name}: {e}")
            return []

    async def stop_container(self, container_name: str) -> bool:
        """Stop and remove container"""
        try:
            if container_name in self.containers:
                container = self.containers[container_name]
                container.stop(timeout=10)
                container.remove()
                del self.containers[container_name]
                logger.info(f"Stopped and removed container: {container_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop container {container_name}: {e}")
            return False

    async def list_containers(self) -> List[Dict[str, Any]]:
        """List all containers"""
        try:
            if not self.client:
                return []

            containers_info = []
            for container in self.client.containers.list(all=True):
                containers_info.append({
                    "id": container.short_id,
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "created": container.attrs.get("Created", ""),
                    "ports": container.ports
                })

            return containers_info
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []


class KubernetesManager:
    """Kubernetes cluster management"""

    def __init__(self):
        self.api_client = None
        self.apps_api = None
        self.core_api = None
        self.deployments = {}

    async def initialize(self, kubeconfig_path: Optional[str] = None) -> bool:
        """Initialize Kubernetes client"""
        try:
            if not KUBERNETES_AVAILABLE:
                logger.warning("Kubernetes library not available")
                return False

            # Load kubeconfig
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()

            # Create API clients
            self.api_client = client.ApiClient()
            self.apps_api = client.AppsV1Api()
            self.core_api = client.CoreV1Api()

            # Test connection
            version = await self._get_cluster_version()
            logger.info(f"Connected to Kubernetes cluster: {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes: {e}")
            return False

    async def _get_cluster_version(self) -> str:
        """Get Kubernetes cluster version"""
        try:
            version_info = self.core_api.get_code()
            return f"{version_info.major}.{version_info.minor}"
        except Exception:
            return "unknown"

    async def deploy_application(self, spec: ContainerSpec, namespace: str = "default") -> DeploymentResult:
        """Deploy application to Kubernetes"""
        try:
            if not self.apps_api:
                return DeploymentResult(False, "", status="failed", logs=["Kubernetes not initialized"])

            deployment_name = f"{spec.name}-deployment"

            # Create deployment manifest
            deployment = self._create_deployment_manifest(spec, namespace)

            # Apply deployment
            try:
                self.apps_api.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment
                )
                operation = "created"
            except client.ApiException as e:
                if e.status == 409:  # Already exists
                    self.apps_api.patch_namespaced_deployment(
                        name=deployment_name,
                        namespace=namespace,
                        body=deployment
                    )
                    operation = "updated"
                else:
                    raise

            # Create service if ports are specified
            service_endpoint = None
            if spec.ports:
                service = self._create_service_manifest(spec, namespace)
                try:
                    self.core_api.create_namespaced_service(
                        namespace=namespace,
                        body=service
                    )
                    service_endpoint = f"{spec.name}-service.{namespace}.svc.cluster.local"
                except client.ApiException as e:
                    if e.status == 409:  # Already exists
                        self.core_api.patch_namespaced_service(
                            name=f"{spec.name}-service",
                            namespace=namespace,
                            body=service
                        )
                        service_endpoint = f"{spec.name}-service.{namespace}.svc.cluster.local"

            self.deployments[deployment_name] = {
                "spec": spec,
                "namespace": namespace,
                "created_at": datetime.now()
            }

            logger.info(f"Successfully {operation} Kubernetes deployment: {deployment_name}")

            return DeploymentResult(
                success=True,
                deployment_id=deployment_name,
                endpoint=service_endpoint,
                status="deployed",
                logs=[f"Deployment {operation} successfully"],
                metadata={
                    "namespace": namespace,
                    "replicas": spec.replicas,
                    "operation": operation
                }
            )

        except Exception as e:
            logger.error(f"Failed to deploy to Kubernetes: {e}")
            return DeploymentResult(
                success=False,
                deployment_id="",
                status="failed",
                logs=[f"Deployment failed: {str(e)}"]
            )

    def _create_deployment_manifest(self, spec: ContainerSpec, namespace: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{spec.name}-deployment",
                "namespace": namespace,
                "labels": {**spec.labels, "app": spec.name}
            },
            "spec": {
                "replicas": spec.replicas,
                "selector": {
                    "matchLabels": {"app": spec.name}
                },
                "template": {
                    "metadata": {
                        "labels": {**spec.labels, "app": spec.name}
                    },
                    "spec": {
                        "containers": [{
                            "name": spec.name,
                            "image": spec.image,
                            "resources": {
                                "limits": {
                                    "memory": spec.memory_limit,
                                    "cpu": spec.cpu_limit
                                },
                                "requests": {
                                    "memory": spec.memory_limit,
                                    "cpu": spec.cpu_limit
                                }
                            },
                            "env": [
                                {"name": k, "value": v}
                                for k, v in spec.env_vars.items()
                            ]
                        }]
                    }
                }
            }
        }

        # Add command and args if specified
        container_spec = deployment["spec"]["template"]["spec"]["containers"][0]
        if spec.command:
            container_spec["command"] = spec.command
        if spec.args:
            container_spec["args"] = spec.args

        # Add ports if specified
        if spec.ports:
            container_spec["ports"] = [
                {"containerPort": port, "protocol": "TCP"}
                for port in spec.ports
            ]

        return deployment

    def _create_service_manifest(self, spec: ContainerSpec, namespace: str) -> Dict[str, Any]:
        """Create Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{spec.name}-service",
                "namespace": namespace,
                "labels": {**spec.labels, "app": spec.name}
            },
            "spec": {
                "selector": {"app": spec.name},
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": port,
                        "targetPort": port
                    }
                    for port in spec.ports
                ],
                "type": "ClusterIP"
            }
        }

    async def get_deployment_status(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )

            return {
                "name": deployment.metadata.name,
                "namespace": deployment.metadata.namespace,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message
                    }
                    for condition in (deployment.status.conditions or [])
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}

    async def delete_deployment(self, deployment_name: str, namespace: str = "default") -> bool:
        """Delete Kubernetes deployment"""
        try:
            # Delete deployment
            self.apps_api.delete_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )

            # Delete service if exists
            service_name = deployment_name.replace("-deployment", "-service")
            try:
                self.core_api.delete_namespaced_service(
                    name=service_name,
                    namespace=namespace
                )
            except client.ApiException:
                pass  # Service might not exist

            if deployment_name in self.deployments:
                del self.deployments[deployment_name]

            logger.info(f"Deleted Kubernetes deployment: {deployment_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete deployment {deployment_name}: {e}")
            return False


class AWSManager:
    """Amazon Web Services integration"""

    def __init__(self):
        self.session = None
        self.ec2_client = None
        self.ecs_client = None
        self.lambda_client = None
        self.resources = {}

    async def initialize(self, aws_access_key: Optional[str] = None,
                        aws_secret_key: Optional[str] = None,
                        region: str = "us-east-1") -> bool:
        """Initialize AWS clients"""
        try:
            if not AWS_AVAILABLE:
                logger.warning("AWS SDK not available")
                return False

            # Create session
            if aws_access_key and aws_secret_key:
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region
                )
            else:
                # Use default credentials
                self.session = boto3.Session(region_name=region)

            # Create service clients
            self.ec2_client = self.session.client('ec2')
            self.ecs_client = self.session.client('ecs')
            self.lambda_client = self.session.client('lambda')

            # Test connection
            self.ec2_client.describe_regions()

            logger.info("AWS clients initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AWS: {e}")
            return False

    async def launch_ec2_instance(self, instance_type: str = "t3.micro",
                                 image_id: str = "ami-0c02fb55956c7d316") -> Optional[str]:
        """Launch EC2 instance"""
        try:
            if not self.ec2_client:
                return None

            response = self.ec2_client.run_instances(
                ImageId=image_id,
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_type,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'Terminal-Coder-Instance'},
                            {'Key': 'CreatedBy', 'Value': 'Terminal-Coder'}
                        ]
                    }
                ]
            )

            instance_id = response['Instances'][0]['InstanceId']
            self.resources[instance_id] = {
                "type": "ec2_instance",
                "created_at": datetime.now(),
                "instance_type": instance_type
            }

            logger.info(f"Launched EC2 instance: {instance_id}")
            return instance_id

        except Exception as e:
            logger.error(f"Failed to launch EC2 instance: {e}")
            return None

    async def deploy_lambda_function(self, function_name: str, code_path: str,
                                   handler: str = "lambda_function.lambda_handler",
                                   runtime: str = "python3.9") -> bool:
        """Deploy AWS Lambda function"""
        try:
            if not self.lambda_client:
                return False

            # Read and zip the code
            with open(code_path, 'rb') as f:
                code_bytes = f.read()

            # Create or update function
            try:
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Role=f"arn:aws:iam::{self._get_account_id()}:role/lambda-execution-role",
                    Handler=handler,
                    Code={'ZipFile': code_bytes},
                    Description='Function deployed by Terminal Coder',
                    Tags={
                        'CreatedBy': 'Terminal-Coder'
                    }
                )
                operation = "created"
            except self.lambda_client.exceptions.ResourceConflictException:
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=code_bytes
                )
                operation = "updated"

            function_arn = response.get('FunctionArn', '')
            self.resources[function_name] = {
                "type": "lambda_function",
                "arn": function_arn,
                "created_at": datetime.now()
            }

            logger.info(f"Successfully {operation} Lambda function: {function_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy Lambda function {function_name}: {e}")
            return False

    def _get_account_id(self) -> str:
        """Get AWS account ID"""
        try:
            sts_client = self.session.client('sts')
            return sts_client.get_caller_identity()['Account']
        except Exception:
            return "123456789012"  # Fallback


class GCPManager:
    """Google Cloud Platform integration"""

    def __init__(self):
        self.credentials = None
        self.project_id = None
        self.compute_client = None
        self.container_client = None
        self.resources = {}

    async def initialize(self, service_account_path: Optional[str] = None,
                        project_id: Optional[str] = None) -> bool:
        """Initialize GCP clients"""
        try:
            if not GCP_AVAILABLE:
                logger.warning("GCP SDK not available")
                return False

            # Set up credentials
            if service_account_path and Path(service_account_path).exists():
                self.credentials = service_account.Credentials.from_service_account_file(
                    service_account_path
                )
            else:
                # Use default credentials
                os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', '')

            # Set project ID
            self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
            if not self.project_id:
                logger.error("GCP project ID not specified")
                return False

            # Create service clients
            self.compute_client = compute_v1.InstancesClient(credentials=self.credentials)
            self.container_client = container_v1.ClusterManagerClient(credentials=self.credentials)

            logger.info(f"GCP clients initialized for project: {self.project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GCP: {e}")
            return False

    async def create_compute_instance(self, instance_name: str, zone: str = "us-central1-a",
                                    machine_type: str = "e2-micro") -> Optional[str]:
        """Create GCP Compute Engine instance"""
        try:
            if not self.compute_client:
                return None

            # Instance configuration
            instance = {
                "name": instance_name,
                "machine_type": f"zones/{zone}/machineTypes/{machine_type}",
                "disks": [
                    {
                        "boot": True,
                        "auto_delete": True,
                        "initialize_params": {
                            "disk_size_gb": "20",
                            "source_image": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts"
                        }
                    }
                ],
                "network_interfaces": [
                    {
                        "network": f"projects/{self.project_id}/global/networks/default",
                        "access_configs": [
                            {
                                "type": "ONE_TO_ONE_NAT",
                                "name": "External NAT"
                            }
                        ]
                    }
                ],
                "labels": {
                    "created-by": "terminal-coder"
                }
            }

            operation = self.compute_client.insert(
                project=self.project_id,
                zone=zone,
                instance_resource=instance
            )

            # Wait for operation to complete (simplified)
            instance_url = f"projects/{self.project_id}/zones/{zone}/instances/{instance_name}"
            self.resources[instance_name] = {
                "type": "compute_instance",
                "zone": zone,
                "machine_type": machine_type,
                "created_at": datetime.now()
            }

            logger.info(f"Created GCP Compute instance: {instance_name}")
            return instance_url

        except Exception as e:
            logger.error(f"Failed to create GCP instance {instance_name}: {e}")
            return None


class CloudManager:
    """Unified cloud management system"""

    def __init__(self):
        self.docker = DockerManager()
        self.kubernetes = KubernetesManager()
        self.aws = AWSManager()
        self.gcp = GCPManager()

        self.active_deployments = {}
        self.deployment_history = []

    async def initialize_cloud_providers(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Initialize all configured cloud providers"""
        results = {}

        # Initialize Docker
        if config.get("docker", {}).get("enabled", False):
            results["docker"] = await self.docker.initialize()

        # Initialize Kubernetes
        if config.get("kubernetes", {}).get("enabled", False):
            kubeconfig = config["kubernetes"].get("kubeconfig_path")
            results["kubernetes"] = await self.kubernetes.initialize(kubeconfig)

        # Initialize AWS
        if config.get("aws", {}).get("enabled", False):
            aws_config = config["aws"]
            results["aws"] = await self.aws.initialize(
                aws_config.get("access_key"),
                aws_config.get("secret_key"),
                aws_config.get("region", "us-east-1")
            )

        # Initialize GCP
        if config.get("gcp", {}).get("enabled", False):
            gcp_config = config["gcp"]
            results["gcp"] = await self.gcp.initialize(
                gcp_config.get("service_account_path"),
                gcp_config.get("project_id")
            )

        active_providers = [provider for provider, status in results.items() if status]
        logger.info(f"Initialized cloud providers: {active_providers}")

        return results

    async def containerize_and_deploy(self, code_path: str, deployment_target: str,
                                    container_spec: ContainerSpec) -> DeploymentResult:
        """Containerize code and deploy to specified target"""
        try:
            deployment_id = f"deploy_{int(time.time())}"

            # Generate Dockerfile if needed
            dockerfile_path = await self._generate_dockerfile(code_path, container_spec)

            # Build container image
            image_name = f"{container_spec.name}:latest"
            if not await self.docker.build_image(dockerfile_path, image_name):
                return DeploymentResult(
                    success=False,
                    deployment_id=deployment_id,
                    status="build_failed",
                    logs=["Failed to build Docker image"]
                )

            container_spec.image = image_name

            # Deploy based on target
            if deployment_target == "docker":
                container_id = await self.docker.run_container(container_spec)
                if container_id:
                    result = DeploymentResult(
                        success=True,
                        deployment_id=container_id,
                        status="running",
                        logs=["Container started successfully"],
                        metadata={"provider": "docker", "container_id": container_id}
                    )
                else:
                    result = DeploymentResult(
                        success=False,
                        deployment_id=deployment_id,
                        status="failed",
                        logs=["Failed to start container"]
                    )

            elif deployment_target == "kubernetes":
                result = await self.kubernetes.deploy_application(container_spec)

            elif deployment_target == "aws":
                # Deploy to AWS ECS or similar service
                result = await self._deploy_to_aws(container_spec)

            elif deployment_target == "gcp":
                # Deploy to GCP Cloud Run or similar service
                result = await self._deploy_to_gcp(container_spec)

            else:
                result = DeploymentResult(
                    success=False,
                    deployment_id=deployment_id,
                    status="failed",
                    logs=[f"Unknown deployment target: {deployment_target}"]
                )

            # Record deployment
            if result.success:
                self.active_deployments[result.deployment_id] = {
                    "spec": container_spec,
                    "target": deployment_target,
                    "created_at": datetime.now(),
                    "status": result.status
                }

            self.deployment_history.append({
                "deployment_id": result.deployment_id,
                "target": deployment_target,
                "success": result.success,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                status="failed",
                logs=[f"Deployment error: {str(e)}"]
            )

    async def _generate_dockerfile(self, code_path: str, spec: ContainerSpec) -> str:
        """Generate Dockerfile for the code"""
        try:
            code_dir = Path(code_path).parent
            dockerfile_path = code_dir / "Dockerfile"

            # Detect language and generate appropriate Dockerfile
            if any(Path(code_dir).glob("*.py")):
                dockerfile_content = self._generate_python_dockerfile(spec)
            elif any(Path(code_dir).glob("*.js")) or (code_dir / "package.json").exists():
                dockerfile_content = self._generate_nodejs_dockerfile(spec)
            elif any(Path(code_dir).glob("*.go")):
                dockerfile_content = self._generate_go_dockerfile(spec)
            elif any(Path(code_dir).glob("*.rs")) or (code_dir / "Cargo.toml").exists():
                dockerfile_content = self._generate_rust_dockerfile(spec)
            else:
                # Generic Dockerfile
                dockerfile_content = self._generate_generic_dockerfile(spec)

            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)

            logger.info(f"Generated Dockerfile: {dockerfile_path}")
            return str(dockerfile_path)

        except Exception as e:
            logger.error(f"Failed to generate Dockerfile: {e}")
            raise

    def _generate_python_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate Python Dockerfile"""
        return f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
{chr(10).join(f'ENV {k}="{v}"' for k, v in spec.env_vars.items())}

# Expose ports
{chr(10).join(f'EXPOSE {port}' for port in spec.ports)}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:{spec.ports[0] if spec.ports else 8000}/health')" || exit 1

# Run the application
CMD {json.dumps(spec.command or ["python", "-m", "main"])}
"""

    def _generate_nodejs_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate Node.js Dockerfile"""
        return f"""FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \\
    adduser -S appuser -u 1001 -G nodejs && \\
    chown -R appuser:nodejs /app
USER appuser

# Set environment variables
{chr(10).join(f'ENV {k}="{v}"' for k, v in spec.env_vars.items())}

# Expose ports
{chr(10).join(f'EXPOSE {port}' for port in spec.ports)}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{spec.ports[0] if spec.ports else 3000}/health || exit 1

# Run the application
CMD {json.dumps(spec.command or ["npm", "start"])}
"""

    def _generate_go_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate Go Dockerfile"""
        return f"""FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/

# Copy binary from builder stage
COPY --from=builder /app/main .

# Set environment variables
{chr(10).join(f'ENV {k}="{v}"' for k, v in spec.env_vars.items())}

# Expose ports
{chr(10).join(f'EXPOSE {port}' for port in spec.ports)}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD ./main --health-check || exit 1

# Run the application
CMD {json.dumps(spec.command or ["./main"])}
"""

    def _generate_rust_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate Rust Dockerfile"""
        return f"""FROM rust:1.75 as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Build dependencies (cached layer)
RUN mkdir src && echo "fn main() {{}}" > src/main.rs
RUN cargo build --release
RUN rm src/main.rs

# Copy source code and build
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/{spec.name} ./app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
{chr(10).join(f'ENV {k}="{v}"' for k, v in spec.env_vars.items())}

# Expose ports
{chr(10).join(f'EXPOSE {port}' for port in spec.ports)}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD ./app --health || exit 1

# Run the application
CMD {json.dumps(spec.command or ["./app"])}
"""

    def _generate_generic_dockerfile(self, spec: ContainerSpec) -> str:
        """Generate generic Dockerfile"""
        return f"""FROM ubuntu:22.04

WORKDIR /app

# Install basic dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
{chr(10).join(f'ENV {k}="{v}"' for k, v in spec.env_vars.items())}

# Expose ports
{chr(10).join(f'EXPOSE {port}' for port in spec.ports)}

# Run the application
CMD {json.dumps(spec.command or ["bash"])}
"""

    async def _deploy_to_aws(self, spec: ContainerSpec) -> DeploymentResult:
        """Deploy to AWS (simplified)"""
        try:
            # This would typically deploy to ECS, Lambda, or other AWS services
            logger.info(f"AWS deployment not fully implemented for {spec.name}")
            return DeploymentResult(
                success=False,
                deployment_id="aws_placeholder",
                status="not_implemented",
                logs=["AWS deployment not fully implemented"]
            )
        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id="",
                status="failed",
                logs=[f"AWS deployment error: {str(e)}"]
            )

    async def _deploy_to_gcp(self, spec: ContainerSpec) -> DeploymentResult:
        """Deploy to GCP (simplified)"""
        try:
            # This would typically deploy to Cloud Run, GKE, or other GCP services
            logger.info(f"GCP deployment not fully implemented for {spec.name}")
            return DeploymentResult(
                success=False,
                deployment_id="gcp_placeholder",
                status="not_implemented",
                logs=["GCP deployment not fully implemented"]
            )
        except Exception as e:
            logger.error(f"GCP deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id="",
                status="failed",
                logs=[f"GCP deployment error: {str(e)}"]
            )

    async def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment to specified number of replicas"""
        try:
            if deployment_id not in self.active_deployments:
                logger.error(f"Deployment {deployment_id} not found")
                return False

            deployment = self.active_deployments[deployment_id]
            target = deployment["target"]

            if target == "kubernetes":
                # Scale Kubernetes deployment
                deployment_name = deployment_id
                namespace = "default"

                # Get current deployment
                apps_api = self.kubernetes.apps_api
                current_deployment = apps_api.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )

                # Update replicas
                current_deployment.spec.replicas = replicas
                apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=current_deployment
                )

                logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")
                return True

            elif target == "docker":
                # For Docker, we'd need to run additional containers
                logger.warning("Docker scaling not implemented - use orchestration platform")
                return False

            else:
                logger.error(f"Scaling not supported for target: {target}")
                return False

        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_id}: {e}")
            return False

    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment performance metrics"""
        try:
            if deployment_id not in self.active_deployments:
                return {}

            deployment = self.active_deployments[deployment_id]
            metrics = {
                "deployment_id": deployment_id,
                "target": deployment["target"],
                "created_at": deployment["created_at"].isoformat(),
                "status": deployment["status"],
                "uptime": (datetime.now() - deployment["created_at"]).total_seconds()
            }

            # Add target-specific metrics
            if deployment["target"] == "kubernetes":
                k8s_status = await self.kubernetes.get_deployment_status(deployment_id)
                metrics.update(k8s_status)

            elif deployment["target"] == "docker":
                container_logs = await self.docker.get_container_logs(deployment_id, tail=10)
                metrics["recent_logs"] = container_logs

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for {deployment_id}: {e}")
            return {"error": str(e)}

    async def cleanup_deployment(self, deployment_id: str) -> bool:
        """Clean up deployment resources"""
        try:
            if deployment_id not in self.active_deployments:
                logger.warning(f"Deployment {deployment_id} not found in active deployments")
                return False

            deployment = self.active_deployments[deployment_id]
            target = deployment["target"]

            success = False
            if target == "kubernetes":
                success = await self.kubernetes.delete_deployment(deployment_id)
            elif target == "docker":
                success = await self.docker.stop_container(deployment_id)

            if success:
                del self.active_deployments[deployment_id]
                logger.info(f"Cleaned up deployment: {deployment_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to cleanup deployment {deployment_id}: {e}")
            return False

    async def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments"""
        deployments = []
        for deployment_id, deployment in self.active_deployments.items():
            deployments.append({
                "deployment_id": deployment_id,
                "target": deployment["target"],
                "status": deployment["status"],
                "created_at": deployment["created_at"].isoformat(),
                "spec_name": deployment["spec"].name
            })
        return deployments


# Global cloud manager instance
cloud_manager = None


async def initialize_cloud_computing(config: Dict[str, Any]):
    """Initialize cloud computing capabilities"""
    global cloud_manager
    try:
        cloud_manager = CloudManager()

        # Initialize cloud providers based on configuration
        provider_results = await cloud_manager.initialize_cloud_providers(config)

        active_providers = sum(provider_results.values())
        logger.info(f"Cloud computing initialized with {active_providers} active providers")

        return cloud_manager

    except Exception as e:
        logger.error(f"Failed to initialize cloud computing: {e}")
        return None


def get_cloud_manager():
    """Get global cloud manager instance"""
    return cloud_manager