"""
Multi-Modal AI Integration Module
Advanced vision, audio, and multi-sensory AI capabilities for Terminal Coder
"""

import asyncio
import aiohttp
import base64
import io
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageGrab
import cv2
import speech_recognition as sr
import pyttsx3
import pyaudio
import wave
from datetime import datetime
import subprocess
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class MultiModalResponse:
    """Multi-modal AI response with rich media support"""
    content: str
    modality: str  # text, vision, audio, mixed
    confidence: float
    media_data: Dict[str, Any]
    processing_time: float
    model_used: str
    tokens_used: int = 0
    cost: float = 0.0


class ScreenAnalyzer:
    """Advanced screen analysis and UI understanding"""

    def __init__(self):
        self.screenshot_cache = {}
        self.ui_elements_cache = {}

    async def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture screenshot with optional region"""
        try:
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()

            # Convert to numpy array for processing
            screenshot_np = np.array(screenshot)

            # Cache for analysis
            timestamp = datetime.now().isoformat()
            self.screenshot_cache[timestamp] = screenshot_np

            return screenshot_np
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None

    async def analyze_ui_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze UI elements using computer vision"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect buttons using template matching and contours
            buttons = await self._detect_buttons(gray)

            # Detect text regions using OCR
            text_regions = await self._detect_text_regions(image)

            # Detect windows and panels
            windows = await self._detect_windows(gray)

            # Detect input fields
            input_fields = await self._detect_input_fields(gray)

            ui_analysis = {
                "buttons": buttons,
                "text_regions": text_regions,
                "windows": windows,
                "input_fields": input_fields,
                "image_dimensions": image.shape[:2],
                "analysis_timestamp": datetime.now().isoformat()
            }

            return ui_analysis

        except Exception as e:
            logger.error(f"UI analysis failed: {e}")
            return {}

    async def _detect_buttons(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect button elements in the image"""
        try:
            # Use edge detection to find rectangular shapes
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            buttons = []
            for contour in contours:
                # Filter by area and aspect ratio
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Button-like size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    if 0.3 < aspect_ratio < 5.0:  # Button-like aspect ratio
                        buttons.append({
                            "type": "button",
                            "bbox": [x, y, w, h],
                            "area": area,
                            "confidence": 0.7
                        })

            return buttons[:20]  # Limit to top 20 candidates
        except Exception as e:
            logger.error(f"Button detection failed: {e}")
            return []

    async def _detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions using OCR"""
        try:
            # Use pytesseract if available, otherwise basic text detection
            try:
                import pytesseract

                # Get text with bounding boxes
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                text_regions = []

                for i, text in enumerate(data['text']):
                    if text.strip():
                        confidence = int(data['conf'][i])
                        if confidence > 30:  # Minimum confidence threshold
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            text_regions.append({
                                "type": "text",
                                "text": text.strip(),
                                "bbox": [x, y, w, h],
                                "confidence": confidence / 100.0
                            })

                return text_regions

            except ImportError:
                # Fallback: basic text region detection using morphological operations
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # Create kernel for text detection
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                # Apply morphological operations
                morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

                # Find contours
                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                text_regions = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum area for text
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h

                        if aspect_ratio > 1.5:  # Text-like aspect ratio
                            text_regions.append({
                                "type": "text",
                                "text": "[OCR not available]",
                                "bbox": [x, y, w, h],
                                "confidence": 0.5
                            })

                return text_regions[:30]  # Limit results

        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []

    async def _detect_windows(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect window boundaries"""
        try:
            # Use Hough line detection to find window edges
            edges = cv2.Canny(gray_image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

            windows = []
            if lines is not None:
                # Group lines to form rectangles (simplified approach)
                for line in lines[:10]:  # Limit processing
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                    windows.append({
                        "type": "window_edge",
                        "line": [x1, y1, x2, y2],
                        "length": length,
                        "confidence": 0.6
                    })

            return windows
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            return []

    async def _detect_input_fields(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect input fields and text boxes"""
        try:
            # Look for rectangular shapes with specific characteristics
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            input_fields = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Input field characteristics: wider than tall, moderate size
                aspect_ratio = w / h
                area = w * h

                if aspect_ratio > 2 and 500 < area < 20000:
                    input_fields.append({
                        "type": "input_field",
                        "bbox": [x, y, w, h],
                        "area": area,
                        "confidence": 0.6
                    })

            return input_fields[:10]  # Limit results
        except Exception as e:
            logger.error(f"Input field detection failed: {e}")
            return []


class VoiceAI:
    """Advanced voice recognition and synthesis"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.audio_buffer = []

        # Configure TTS
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.8)  # Volume level

        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break

    async def listen_continuously(self, callback=None, duration: int = 30) -> List[str]:
        """Continuously listen for voice commands"""
        recognized_texts = []

        try:
            # Calibrate microphone for ambient noise
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)

            logger.info(f"Listening for {duration} seconds...")

            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < duration:
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                    # Recognize speech
                    text = await self._recognize_speech(audio)
                    if text:
                        recognized_texts.append(text)
                        logger.info(f"Recognized: {text}")

                        if callback:
                            await callback(text)

                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"Speech recognition error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Voice listening failed: {e}")

        return recognized_texts

    async def _recognize_speech(self, audio) -> Optional[str]:
        """Recognize speech from audio using multiple engines"""
        try:
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.debug(f"Google Speech Recognition error: {e}")

            # Fallback to offline recognition if available
            try:
                text = self.recognizer.recognize_sphinx(audio)
                return text
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.debug(f"Sphinx recognition error: {e}")

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")

        return None

    async def speak(self, text: str, async_mode: bool = True) -> bool:
        """Convert text to speech with advanced options"""
        try:
            if async_mode:
                # Run TTS in background
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._speak_sync, text)
            else:
                self._speak_sync(text)
            return True
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return False

    def _speak_sync(self, text: str):
        """Synchronous TTS execution"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    async def record_audio(self, duration: int = 5, filename: Optional[str] = None) -> str:
        """Record audio to file"""
        try:
            if not filename:
                filename = f"/tmp/claude/audio_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

            # Audio recording parameters
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = 44100

            p = pyaudio.PyAudio()

            # Start recording
            stream = p.open(format=format,
                          channels=channels,
                          rate=rate,
                          input=True,
                          frames_per_buffer=chunk)

            logger.info(f"Recording audio for {duration} seconds...")
            frames = []

            for _ in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)

            # Stop recording
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save to file
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            logger.info(f"Audio recorded to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return ""


class MultiModalAI:
    """Advanced multi-modal AI integration with vision and voice"""

    def __init__(self, ai_manager):
        self.ai_manager = ai_manager
        self.screen_analyzer = ScreenAnalyzer()
        self.voice_ai = VoiceAI()
        self.processing_queue = asyncio.Queue()
        self.active_sessions = {}

    async def analyze_screen_with_ai(self, query: str, region: Optional[Tuple] = None) -> MultiModalResponse:
        """Analyze screen content using AI vision models"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Capture screen
            screenshot = await self.screen_analyzer.capture_screen(region)
            if screenshot is None:
                raise Exception("Failed to capture screen")

            # Convert to base64 for AI APIs
            pil_image = Image.fromarray(screenshot)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Prepare messages for vision-capable models
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            # Use vision-capable model
            if self.ai_manager.current_provider == "openai":
                # Use GPT-4 Vision
                response = await self.ai_manager.chat(messages, model="gpt-4-vision-preview")
            elif self.ai_manager.current_provider == "anthropic":
                # Use Claude 3 with vision
                response = await self.ai_manager.chat(messages, model="claude-3-sonnet-20240229")
            elif self.ai_manager.current_provider == "google":
                # Use Gemini Pro Vision
                response = await self.ai_manager.chat(messages, model="gemini-pro-vision")
            else:
                raise Exception("Current provider doesn't support vision")

            processing_time = asyncio.get_event_loop().time() - start_time

            return MultiModalResponse(
                content=response.content,
                modality="vision",
                confidence=0.9,
                media_data={"screenshot_shape": screenshot.shape, "query": query},
                processing_time=processing_time,
                model_used=response.model,
                tokens_used=response.tokens_used,
                cost=response.cost
            )

        except Exception as e:
            logger.error(f"Screen analysis failed: {e}")
            return MultiModalResponse(
                content=f"Failed to analyze screen: {e}",
                modality="error",
                confidence=0.0,
                media_data={},
                processing_time=0.0,
                model_used="none",
                tokens_used=0,
                cost=0.0
            )

    async def voice_to_code(self, duration: int = 10) -> MultiModalResponse:
        """Convert voice commands to code using AI"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Record voice
            logger.info("Starting voice recording...")
            await self.voice_ai.speak("Please speak your coding request now.")

            # Listen for voice input
            voice_texts = await self.voice_ai.listen_continuously(duration=duration)

            if not voice_texts:
                raise Exception("No voice input detected")

            # Combine all recognized text
            combined_text = " ".join(voice_texts)
            logger.info(f"Combined voice input: {combined_text}")

            # Convert to code using AI
            code_prompt = f"""Convert this voice command to working code:

Voice Command: "{combined_text}"

Please generate clean, working code based on this request. Include comments explaining the code."""

            messages = [{"role": "user", "content": code_prompt}]
            response = await self.ai_manager.chat(messages)

            processing_time = asyncio.get_event_loop().time() - start_time

            # Optionally speak the result
            await self.voice_ai.speak("Code generated successfully. Check the output.")

            return MultiModalResponse(
                content=response.content,
                modality="voice",
                confidence=0.8,
                media_data={"voice_input": combined_text, "voice_segments": voice_texts},
                processing_time=processing_time,
                model_used=response.model,
                tokens_used=response.tokens_used,
                cost=response.cost
            )

        except Exception as e:
            logger.error(f"Voice to code failed: {e}")
            await self.voice_ai.speak(f"Voice to code failed: {str(e)}")
            return MultiModalResponse(
                content=f"Failed to convert voice to code: {e}",
                modality="error",
                confidence=0.0,
                media_data={},
                processing_time=0.0,
                model_used="none"
            )

    async def analyze_code_visually(self, image_path: str, analysis_type: str = "bugs") -> MultiModalResponse:
        """Analyze code screenshots for bugs, improvements, or explanations"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, 'rb') as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode()

            # Create analysis prompt based on type
            prompts = {
                "bugs": "Analyze this code image for potential bugs, errors, or issues. Provide specific line-by-line feedback.",
                "improvements": "Analyze this code image and suggest improvements, optimizations, or best practices.",
                "explanation": "Explain what this code does, break down its functionality, and describe its purpose.",
                "security": "Analyze this code image for security vulnerabilities, potential exploits, or unsafe practices.",
                "performance": "Analyze this code image for performance issues and suggest optimizations."
            }

            prompt = prompts.get(analysis_type, prompts["bugs"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            response = await self.ai_manager.chat(messages)
            processing_time = asyncio.get_event_loop().time() - start_time

            return MultiModalResponse(
                content=response.content,
                modality="vision",
                confidence=0.9,
                media_data={"image_path": image_path, "analysis_type": analysis_type},
                processing_time=processing_time,
                model_used=response.model,
                tokens_used=response.tokens_used,
                cost=response.cost
            )

        except Exception as e:
            logger.error(f"Visual code analysis failed: {e}")
            return MultiModalResponse(
                content=f"Failed to analyze code visually: {e}",
                modality="error",
                confidence=0.0,
                media_data={},
                processing_time=0.0,
                model_used="none"
            )

    async def interactive_debugging(self, error_message: str, screenshot: bool = True) -> MultiModalResponse:
        """Interactive debugging with visual context"""
        try:
            start_time = asyncio.get_event_loop().time()

            debug_data = {"error_message": error_message}

            # Optionally capture screen for context
            if screenshot:
                screen_image = await self.screen_analyzer.capture_screen()
                if screen_image is not None:
                    # Analyze UI elements
                    ui_analysis = await self.screen_analyzer.analyze_ui_elements(screen_image)
                    debug_data["ui_context"] = ui_analysis

                    # Convert screenshot for AI
                    pil_image = Image.fromarray(screen_image)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Help debug this error with visual context:

Error: {error_message}

Please analyze the screen and provide debugging suggestions, potential causes, and solutions."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ]
                else:
                    # Text-only debugging
                    messages = [
                        {
                            "role": "user",
                            "content": f"""Help debug this error:

Error: {error_message}

Please provide debugging suggestions, potential causes, and solutions."""
                        }
                    ]
            else:
                # Text-only debugging
                messages = [
                    {
                        "role": "user",
                        "content": f"Help debug this error: {error_message}"
                    }
                ]

            response = await self.ai_manager.chat(messages)
            processing_time = asyncio.get_event_loop().time() - start_time

            return MultiModalResponse(
                content=response.content,
                modality="mixed" if screenshot else "text",
                confidence=0.8,
                media_data=debug_data,
                processing_time=processing_time,
                model_used=response.model,
                tokens_used=response.tokens_used,
                cost=response.cost
            )

        except Exception as e:
            logger.error(f"Interactive debugging failed: {e}")
            return MultiModalResponse(
                content=f"Debugging failed: {e}",
                modality="error",
                confidence=0.0,
                media_data={},
                processing_time=0.0,
                model_used="none"
            )

    async def start_voice_assistant(self):
        """Start continuous voice assistant mode"""
        try:
            logger.info("Starting voice assistant mode...")
            await self.voice_ai.speak("Voice assistant activated. Say 'stop assistant' to exit.")

            while True:
                try:
                    # Listen for commands
                    voice_texts = await self.voice_ai.listen_continuously(duration=5)

                    if not voice_texts:
                        continue

                    command = " ".join(voice_texts).lower()

                    # Check for exit command
                    if "stop assistant" in command or "exit voice" in command:
                        await self.voice_ai.speak("Voice assistant deactivated.")
                        break

                    # Process voice command
                    if "analyze screen" in command:
                        await self.voice_ai.speak("Analyzing screen...")
                        result = await self.analyze_screen_with_ai(command)
                        await self.voice_ai.speak("Screen analysis complete.")

                    elif "generate code" in command or "write code" in command:
                        result = await self.voice_to_code(duration=10)

                    elif "take screenshot" in command:
                        screenshot = await self.screen_analyzer.capture_screen()
                        if screenshot is not None:
                            await self.voice_ai.speak("Screenshot captured successfully.")
                        else:
                            await self.voice_ai.speak("Failed to capture screenshot.")

                    else:
                        # General AI query
                        messages = [{"role": "user", "content": command}]
                        response = await self.ai_manager.chat(messages)

                        # Speak response (limit length)
                        response_text = response.content[:200] + "..." if len(response.content) > 200 else response.content
                        await self.voice_ai.speak(response_text)

                except KeyboardInterrupt:
                    await self.voice_ai.speak("Voice assistant stopped.")
                    break
                except Exception as e:
                    logger.error(f"Voice assistant error: {e}")
                    await self.voice_ai.speak("Sorry, I encountered an error.")

        except Exception as e:
            logger.error(f"Voice assistant failed to start: {e}")


# Global multimodal instance
multimodal_ai = None


async def initialize_multimodal_ai(ai_manager):
    """Initialize multimodal AI capabilities"""
    global multimodal_ai
    try:
        multimodal_ai = MultiModalAI(ai_manager)
        logger.info("Multi-modal AI initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize multi-modal AI: {e}")
        return False


def get_multimodal_ai():
    """Get global multimodal AI instance"""
    return multimodal_ai