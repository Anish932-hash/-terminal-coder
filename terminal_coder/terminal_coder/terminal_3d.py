"""
Advanced 3D Terminal Visualization System
Immersive 3D interfaces, holographic displays, and spatial programming environments
"""

import asyncio
import numpy as np
import time
import logging
import math
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import os

try:
    import moderngl
    import pygame
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import glfw
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    logging.warning("OpenGL libraries not available. 3D features limited.")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """3D view modes"""
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"
    TOP_DOWN = "top_down"
    CODE_MATRIX = "code_matrix"
    HOLOGRAPHIC = "holographic"
    VR_MODE = "vr_mode"


@dataclass
class Vector3D:
    """3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x/mag, self.y/mag, self.z/mag)
        return Vector3D()

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass
class Transform3D:
    """3D transformation matrix"""
    position: Vector3D = field(default_factory=Vector3D)
    rotation: Vector3D = field(default_factory=Vector3D)
    scale: Vector3D = field(default_factory=lambda: Vector3D(1, 1, 1))

    def matrix(self) -> np.ndarray:
        """Get transformation matrix"""
        # Translation matrix
        T = np.array([
            [1, 0, 0, self.position.x],
            [0, 1, 0, self.position.y],
            [0, 0, 1, self.position.z],
            [0, 0, 0, 1]
        ])

        # Rotation matrices
        rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(self.rotation.x), -math.sin(self.rotation.x), 0],
            [0, math.sin(self.rotation.x), math.cos(self.rotation.x), 0],
            [0, 0, 0, 1]
        ])

        ry = np.array([
            [math.cos(self.rotation.y), 0, math.sin(self.rotation.y), 0],
            [0, 1, 0, 0],
            [-math.sin(self.rotation.y), 0, math.cos(self.rotation.y), 0],
            [0, 0, 0, 1]
        ])

        rz = np.array([
            [math.cos(self.rotation.z), -math.sin(self.rotation.z), 0, 0],
            [math.sin(self.rotation.z), math.cos(self.rotation.z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Scale matrix
        S = np.array([
            [self.scale.x, 0, 0, 0],
            [0, self.scale.y, 0, 0],
            [0, 0, self.scale.z, 0],
            [0, 0, 0, 1]
        ])

        return T @ rx @ ry @ rz @ S


@dataclass
class Material:
    """3D material properties"""
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular: float = 0.5
    shininess: float = 32.0
    wireframe: bool = False
    glow: bool = False
    holographic: bool = False


@dataclass
class CodeBlock3D:
    """3D representation of code blocks"""
    code: str
    language: str
    transform: Transform3D = field(default_factory=Transform3D)
    material: Material = field(default_factory=Material)
    connections: List[str] = field(default_factory=list)  # Connected code blocks
    metadata: Dict[str, Any] = field(default_factory=dict)
    animation_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Camera3D:
    """3D camera for viewing the scene"""
    position: Vector3D = field(default_factory=lambda: Vector3D(0, 0, 5))
    target: Vector3D = field(default_factory=Vector3D)
    up: Vector3D = field(default_factory=lambda: Vector3D(0, 1, 0))
    fov: float = 60.0
    near_clip: float = 0.1
    far_clip: float = 1000.0
    view_mode: ViewMode = ViewMode.THIRD_PERSON

    def view_matrix(self) -> np.ndarray:
        """Get view matrix"""
        forward = (self.target - self.position).normalize()
        right = forward.cross(self.up).normalize()
        up = right.cross(forward)

        return np.array([
            [right.x, up.x, -forward.x, 0],
            [right.y, up.y, -forward.y, 0],
            [right.z, up.z, -forward.z, 0],
            [-right.dot(self.position), -up.dot(self.position), forward.dot(self.position), 1]
        ])

    def projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get projection matrix"""
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        return np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far_clip + self.near_clip) / (self.near_clip - self.far_clip),
             (2 * self.far_clip * self.near_clip) / (self.near_clip - self.far_clip)],
            [0, 0, -1, 0]
        ])


class Renderer3D:
    """Advanced 3D renderer for terminal visualization"""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.window = None
        self.context = None
        self.shader_program = None
        self.initialized = False

        # Rendering state
        self.camera = Camera3D()
        self.objects = {}
        self.lights = []
        self.effects = {}

        # Animation system
        self.animations = {}
        self.time = 0.0

        # Shaders
        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = normal_matrix * aNormal;
            TexCoord = aTexCoord;

            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """

        self.fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;

        uniform vec3 viewPos;
        uniform vec4 objectColor;
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform bool holographic;
        uniform bool glow;
        uniform float time;

        void main() {
            vec3 color = objectColor.rgb;

            // Lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // Specular
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = spec * lightColor;

            // Holographic effect
            if (holographic) {
                float hologram = sin(FragPos.y * 20.0 + time * 5.0) * 0.5 + 0.5;
                color = mix(color, vec3(0.0, 1.0, 1.0), hologram * 0.3);
                color.rgb += vec3(0.0, 0.2, 0.4);
            }

            // Glow effect
            if (glow) {
                color.rgb += vec3(0.2, 0.2, 0.8) * sin(time * 3.0) * 0.5 + 0.5;
            }

            vec3 result = (diffuse + specular) * color;
            FragColor = vec4(result, objectColor.a);
        }
        """

    async def initialize(self) -> bool:
        """Initialize 3D rendering system"""
        try:
            if not OPENGL_AVAILABLE:
                logger.warning("OpenGL not available, using ASCII 3D fallback")
                return self._initialize_ascii_3d()

            # Initialize GLFW
            if not glfw.init():
                logger.error("Failed to initialize GLFW")
                return False

            # Create window
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            self.window = glfw.create_window(
                self.width, self.height, "Terminal Coder 3D", None, None
            )

            if not self.window:
                logger.error("Failed to create GLFW window")
                glfw.terminate()
                return False

            glfw.make_context_current(self.window)

            # Initialize OpenGL
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Compile shaders
            if not await self._compile_shaders():
                return False

            # Setup default lighting
            self.lights.append({
                "position": Vector3D(2, 2, 2),
                "color": (1.0, 1.0, 1.0),
                "intensity": 1.0
            })

            self.initialized = True
            logger.info("3D rendering system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize 3D renderer: {e}")
            return self._initialize_ascii_3d()

    def _initialize_ascii_3d(self) -> bool:
        """Initialize ASCII-based 3D fallback"""
        try:
            self.ascii_mode = True
            self.initialized = True
            logger.info("ASCII 3D mode initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ASCII 3D: {e}")
            return False

    async def _compile_shaders(self) -> bool:
        """Compile vertex and fragment shaders"""
        try:
            # Vertex shader
            vertex_shader = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vertex_shader, self.vertex_shader_source)
            glCompileShader(vertex_shader)

            if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(vertex_shader)
                logger.error(f"Vertex shader compilation failed: {error}")
                return False

            # Fragment shader
            fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragment_shader, self.fragment_shader_source)
            glCompileShader(fragment_shader)

            if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(fragment_shader)
                logger.error(f"Fragment shader compilation failed: {error}")
                return False

            # Shader program
            self.shader_program = glCreateProgram()
            glAttachShader(self.shader_program, vertex_shader)
            glAttachShader(self.shader_program, fragment_shader)
            glLinkProgram(self.shader_program)

            if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
                error = glGetProgramInfoLog(self.shader_program)
                logger.error(f"Shader program linking failed: {error}")
                return False

            # Clean up
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            return True

        except Exception as e:
            logger.error(f"Shader compilation failed: {e}")
            return False

    async def add_code_block(self, block_id: str, code_block: CodeBlock3D):
        """Add a 3D code block to the scene"""
        try:
            # Generate geometry for the code block
            vertices, indices = self._generate_code_block_geometry(code_block)

            # Create VAO, VBO, EBO
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            ebo = glGenBuffers(1)

            glBindVertexArray(vao)

            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

            # Position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            # Normal attribute
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)

            # Texture coordinate attribute
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
            glEnableVertexAttribArray(2)

            self.objects[block_id] = {
                "code_block": code_block,
                "vao": vao,
                "vbo": vbo,
                "ebo": ebo,
                "vertex_count": len(indices)
            }

            logger.debug(f"Added 3D code block: {block_id}")

        except Exception as e:
            logger.error(f"Failed to add code block {block_id}: {e}")

    def _generate_code_block_geometry(self, code_block: CodeBlock3D) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D geometry for a code block"""
        # Create a textured cube/plane for the code block
        width = max(1.0, len(code_block.code.split('\n')[0]) * 0.1)
        height = max(0.5, len(code_block.code.split('\n')) * 0.1)
        depth = 0.1

        # Vertices: position (3) + normal (3) + texcoord (2)
        vertices = np.array([
            # Front face
            [-width/2, -height/2,  depth/2,  0, 0, 1,  0, 0],
            [ width/2, -height/2,  depth/2,  0, 0, 1,  1, 0],
            [ width/2,  height/2,  depth/2,  0, 0, 1,  1, 1],
            [-width/2,  height/2,  depth/2,  0, 0, 1,  0, 1],

            # Back face
            [-width/2, -height/2, -depth/2,  0, 0, -1, 1, 0],
            [-width/2,  height/2, -depth/2,  0, 0, -1, 1, 1],
            [ width/2,  height/2, -depth/2,  0, 0, -1, 0, 1],
            [ width/2, -height/2, -depth/2,  0, 0, -1, 0, 0],

            # Left face
            [-width/2, -height/2, -depth/2, -1, 0, 0,  0, 0],
            [-width/2, -height/2,  depth/2, -1, 0, 0,  1, 0],
            [-width/2,  height/2,  depth/2, -1, 0, 0,  1, 1],
            [-width/2,  height/2, -depth/2, -1, 0, 0,  0, 1],

            # Right face
            [ width/2, -height/2, -depth/2,  1, 0, 0,  1, 0],
            [ width/2,  height/2, -depth/2,  1, 0, 0,  1, 1],
            [ width/2,  height/2,  depth/2,  1, 0, 0,  0, 1],
            [ width/2, -height/2,  depth/2,  1, 0, 0,  0, 0],

            # Top face
            [-width/2,  height/2, -depth/2,  0, 1, 0,  0, 1],
            [-width/2,  height/2,  depth/2,  0, 1, 0,  0, 0],
            [ width/2,  height/2,  depth/2,  0, 1, 0,  1, 0],
            [ width/2,  height/2, -depth/2,  0, 1, 0,  1, 1],

            # Bottom face
            [-width/2, -height/2, -depth/2,  0, -1, 0, 1, 1],
            [ width/2, -height/2, -depth/2,  0, -1, 0, 0, 1],
            [ width/2, -height/2,  depth/2,  0, -1, 0, 0, 0],
            [-width/2, -height/2,  depth/2,  0, -1, 0, 1, 0]
        ], dtype=np.float32)

        # Indices for triangles
        indices = np.array([
            0,  1,  2,   2,  3,  0,   # front
            4,  5,  6,   6,  7,  4,   # back
            8,  9,  10,  10, 11, 8,   # left
            12, 13, 14,  14, 15, 12,  # right
            16, 17, 18,  18, 19, 16,  # top
            20, 21, 22,  22, 23, 20   # bottom
        ], dtype=np.uint32)

        return vertices, indices

    async def render_frame(self):
        """Render a complete 3D frame"""
        try:
            if not self.initialized:
                return

            if hasattr(self, 'ascii_mode') and self.ascii_mode:
                return await self._render_ascii_frame()

            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.1, 0.1, 0.2, 1.0)

            # Use shader program
            glUseProgram(self.shader_program)

            # Update time
            self.time += 0.016  # ~60fps

            # Set uniforms
            view_matrix = self.camera.view_matrix()
            projection_matrix = self.camera.projection_matrix(self.width / self.height)

            view_loc = glGetUniformLocation(self.shader_program, "view")
            glUniformMatrix4fv(view_loc, 1, GL_TRUE, view_matrix)

            projection_loc = glGetUniformLocation(self.shader_program, "projection")
            glUniformMatrix4fv(projection_loc, 1, GL_TRUE, projection_matrix)

            # Camera position
            view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
            glUniform3f(view_pos_loc, self.camera.position.x, self.camera.position.y, self.camera.position.z)

            # Light
            light_pos_loc = glGetUniformLocation(self.shader_program, "lightPos")
            light_color_loc = glGetUniformLocation(self.shader_program, "lightColor")

            if self.lights:
                light = self.lights[0]
                glUniform3f(light_pos_loc, light["position"].x, light["position"].y, light["position"].z)
                glUniform3f(light_color_loc, *light["color"])

            # Time uniform
            time_loc = glGetUniformLocation(self.shader_program, "time")
            glUniform1f(time_loc, self.time)

            # Render all objects
            for obj_id, obj_data in self.objects.items():
                await self._render_object(obj_data)

            # Swap buffers
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        except Exception as e:
            logger.error(f"Rendering failed: {e}")

    async def _render_object(self, obj_data):
        """Render a single 3D object"""
        try:
            code_block = obj_data["code_block"]

            # Model matrix
            model_matrix = code_block.transform.matrix()
            model_loc = glGetUniformLocation(self.shader_program, "model")
            glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_matrix)

            # Normal matrix
            normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
            normal_loc = glGetUniformLocation(self.shader_program, "normal_matrix")
            glUniformMatrix3fv(normal_loc, 1, GL_TRUE, normal_matrix)

            # Material properties
            color_loc = glGetUniformLocation(self.shader_program, "objectColor")
            glUniform4f(color_loc, *code_block.material.color)

            holographic_loc = glGetUniformLocation(self.shader_program, "holographic")
            glUniform1i(holographic_loc, code_block.material.holographic)

            glow_loc = glGetUniformLocation(self.shader_program, "glow")
            glUniform1i(glow_loc, code_block.material.glow)

            # Render
            glBindVertexArray(obj_data["vao"])
            glDrawElements(GL_TRIANGLES, obj_data["vertex_count"], GL_UNSIGNED_INT, None)

        except Exception as e:
            logger.error(f"Object rendering failed: {e}")

    async def _render_ascii_frame(self):
        """Render frame using ASCII art (fallback)"""
        try:
            if not RICH_AVAILABLE:
                return

            console = Console()
            console.clear()

            # Create 3D-like ASCII visualization
            layout = Layout()

            # Create a simple 3D ASCII scene
            ascii_scene = self._generate_ascii_3d_scene()

            layout.split_column(
                Layout(Panel(ascii_scene, title="3D Code Visualization"), name="main"),
                Layout(Panel(self._get_camera_info(), title="Camera Info"), name="info", size=5)
            )

            console.print(layout)

        except Exception as e:
            logger.error(f"ASCII rendering failed: {e}")

    def _generate_ascii_3d_scene(self) -> str:
        """Generate ASCII art 3D scene"""
        try:
            scene_lines = []
            width, height = 80, 24

            # Simple isometric projection
            for y in range(height):
                line = ""
                for x in range(width):
                    # Calculate 3D position
                    world_x = (x - width/2) * 0.1
                    world_y = (y - height/2) * 0.1
                    world_z = 0

                    # Check if any objects are at this position
                    char = ' '
                    for obj_id, obj_data in self.objects.items():
                        code_block = obj_data["code_block"]
                        pos = code_block.transform.position

                        # Simple distance check
                        distance = math.sqrt(
                            (world_x - pos.x)**2 +
                            (world_y - pos.y)**2 +
                            (world_z - pos.z)**2
                        )

                        if distance < 0.5:
                            if code_block.material.holographic:
                                char = '░'
                            elif code_block.material.glow:
                                char = '▓'
                            else:
                                char = '█'
                            break

                    line += char

                scene_lines.append(line)

            return "\n".join(scene_lines)

        except Exception as e:
            logger.error(f"ASCII scene generation failed: {e}")
            return "ASCII 3D Scene Generation Failed"

    def _get_camera_info(self) -> str:
        """Get camera information display"""
        return f"""Position: ({self.camera.position.x:.2f}, {self.camera.position.y:.2f}, {self.camera.position.z:.2f})
Target: ({self.camera.target.x:.2f}, {self.camera.target.y:.2f}, {self.camera.target.z:.2f})
Mode: {self.camera.view_mode.value}
Objects: {len(self.objects)}"""

    async def set_camera_mode(self, mode: ViewMode):
        """Set camera view mode"""
        self.camera.view_mode = mode

        if mode == ViewMode.FIRST_PERSON:
            self.camera.position = Vector3D(0, 0, 0)
            self.camera.target = Vector3D(0, 0, -1)
        elif mode == ViewMode.THIRD_PERSON:
            self.camera.position = Vector3D(0, 5, 10)
            self.camera.target = Vector3D(0, 0, 0)
        elif mode == ViewMode.TOP_DOWN:
            self.camera.position = Vector3D(0, 10, 0)
            self.camera.target = Vector3D(0, 0, 0)
            self.camera.up = Vector3D(0, 0, -1)
        elif mode == ViewMode.CODE_MATRIX:
            self.camera.position = Vector3D(0, 0, 20)
            self.camera.target = Vector3D(0, 0, 0)
        elif mode == ViewMode.HOLOGRAPHIC:
            self.camera.position = Vector3D(5, 5, 5)
            self.camera.target = Vector3D(0, 0, 0)

    async def animate_code_flow(self, from_block: str, to_block: str, duration: float = 2.0):
        """Animate data flow between code blocks"""
        try:
            if from_block not in self.objects or to_block not in self.objects:
                return

            from_pos = self.objects[from_block]["code_block"].transform.position
            to_pos = self.objects[to_block]["code_block"].transform.position

            # Create particle animation
            animation_id = f"flow_{from_block}_{to_block}_{time.time()}"

            self.animations[animation_id] = {
                "type": "flow",
                "from": from_pos,
                "to": to_pos,
                "start_time": self.time,
                "duration": duration,
                "particles": self._generate_flow_particles(from_pos, to_pos)
            }

            logger.debug(f"Started flow animation: {animation_id}")

        except Exception as e:
            logger.error(f"Animation failed: {e}")

    def _generate_flow_particles(self, from_pos: Vector3D, to_pos: Vector3D) -> List[Dict]:
        """Generate particles for flow animation"""
        particles = []
        particle_count = 20

        for i in range(particle_count):
            particles.append({
                "position": Vector3D(from_pos.x, from_pos.y, from_pos.z),
                "velocity": Vector3D(0, 0, 0),
                "life": 1.0,
                "size": 0.05,
                "color": (0.0, 1.0, 0.8, 1.0)
            })

        return particles

    async def cleanup(self):
        """Cleanup 3D resources"""
        try:
            # Clean up OpenGL resources
            for obj_data in self.objects.values():
                if "vao" in obj_data:
                    glDeleteVertexArrays(1, [obj_data["vao"]])
                if "vbo" in obj_data:
                    glDeleteBuffers(1, [obj_data["vbo"]])
                if "ebo" in obj_data:
                    glDeleteBuffers(1, [obj_data["ebo"]])

            if self.shader_program:
                glDeleteProgram(self.shader_program)

            if self.window:
                glfw.destroy_window(self.window)
                glfw.terminate()

            logger.info("3D renderer cleaned up successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class Terminal3DManager:
    """Manager for 3D terminal visualization"""

    def __init__(self):
        self.renderer = Renderer3D()
        self.running = False
        self.render_thread = None
        self.code_blocks = {}
        self.interaction_handlers = {}

    async def initialize(self) -> bool:
        """Initialize 3D terminal system"""
        try:
            success = await self.renderer.initialize()
            if success:
                logger.info("3D Terminal Manager initialized successfully")

                # Set up default scene
                await self._setup_default_scene()

            return success
        except Exception as e:
            logger.error(f"Failed to initialize 3D terminal manager: {e}")
            return False

    async def _setup_default_scene(self):
        """Setup default 3D scene"""
        try:
            # Add welcome code block
            welcome_code = """# Welcome to Terminal Coder 3D
print("Hello, 3D World!")
def explore_3d():
    return "Amazing 3D coding experience"
"""

            welcome_block = CodeBlock3D(
                code=welcome_code,
                language="python",
                transform=Transform3D(
                    position=Vector3D(0, 0, 0),
                    scale=Vector3D(1, 1, 1)
                ),
                material=Material(
                    color=(0.2, 0.8, 0.2, 0.9),
                    glow=True,
                    holographic=True
                )
            )

            await self.renderer.add_code_block("welcome", welcome_block)
            self.code_blocks["welcome"] = welcome_block

        except Exception as e:
            logger.error(f"Default scene setup failed: {e}")

    async def add_code_visualization(self, code_id: str, code: str, language: str = "python",
                                   position: Optional[Vector3D] = None,
                                   effects: Optional[Dict] = None):
        """Add code for 3D visualization"""
        try:
            if position is None:
                position = Vector3D(
                    len(self.code_blocks) * 2.0, 0, 0
                )

            material = Material()
            if effects:
                material.glow = effects.get("glow", False)
                material.holographic = effects.get("holographic", False)
                material.color = effects.get("color", (1.0, 1.0, 1.0, 1.0))

            code_block = CodeBlock3D(
                code=code,
                language=language,
                transform=Transform3D(position=position),
                material=material
            )

            await self.renderer.add_code_block(code_id, code_block)
            self.code_blocks[code_id] = code_block

            logger.info(f"Added 3D code visualization: {code_id}")

        except Exception as e:
            logger.error(f"Failed to add code visualization {code_id}: {e}")

    async def start_render_loop(self):
        """Start the 3D rendering loop"""
        try:
            self.running = True
            logger.info("Starting 3D render loop...")

            while self.running:
                await self.renderer.render_frame()
                await asyncio.sleep(0.016)  # ~60 FPS

        except Exception as e:
            logger.error(f"Render loop error: {e}")
        finally:
            self.running = False

    async def set_view_mode(self, mode: str):
        """Set 3D view mode"""
        try:
            view_mode = ViewMode(mode)
            await self.renderer.set_camera_mode(view_mode)
            logger.info(f"Set view mode to: {mode}")
        except ValueError:
            logger.error(f"Invalid view mode: {mode}")

    async def create_code_flow_animation(self, from_code: str, to_code: str):
        """Create animation showing code flow between blocks"""
        try:
            await self.renderer.animate_code_flow(from_code, to_code)
            logger.info(f"Created flow animation: {from_code} -> {to_code}")
        except Exception as e:
            logger.error(f"Flow animation failed: {e}")

    async def enable_holographic_mode(self):
        """Enable holographic display mode"""
        try:
            await self.renderer.set_camera_mode(ViewMode.HOLOGRAPHIC)

            # Apply holographic effects to all code blocks
            for code_block in self.code_blocks.values():
                code_block.material.holographic = True
                code_block.material.glow = True
                code_block.material.color = (0.0, 1.0, 1.0, 0.7)

            logger.info("Holographic mode enabled")

        except Exception as e:
            logger.error(f"Failed to enable holographic mode: {e}")

    async def shutdown(self):
        """Shutdown 3D terminal system"""
        try:
            self.running = False
            await self.renderer.cleanup()
            logger.info("3D terminal system shut down")
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")


# Global 3D terminal instance
terminal_3d = None


async def initialize_terminal_3d():
    """Initialize global 3D terminal system"""
    global terminal_3d
    try:
        terminal_3d = Terminal3DManager()
        success = await terminal_3d.initialize()

        if success:
            logger.info("Terminal 3D system initialized successfully")
        else:
            logger.warning("Terminal 3D system initialized with limitations")

        return terminal_3d
    except Exception as e:
        logger.error(f"Failed to initialize Terminal 3D: {e}")
        return None


def get_terminal_3d():
    """Get global 3D terminal instance"""
    return terminal_3d