"""
Voice service module for the mental health chatbot.
Provides Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.
"""

import os
import logging
import subprocess
import platform
import tempfile
from typing import Optional, Tuple
from io import BytesIO

# Voice-related imports
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq

from dotenv import load_dotenv
from .error_handler import APIError, ConfigurationError

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class VoiceService:
    """Service class for handling voice operations in the mental health chatbot."""
    
    def __init__(self):
        """Initialize the voice service with API keys and configurations."""
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.elevenlabs_api_key = os.getenv('ELEVEN_API_KEY')
        
        # Initialize clients
        self.groq_client = None
        self.elevenlabs_client = None
        
        # Initialize Groq client if API key is available
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # Initialize ElevenLabs client if API key is available
        if self.elevenlabs_api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
                logger.info("ElevenLabs client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ElevenLabs client: {e}")
    
    def text_to_speech_gtts(self, text: str, output_filepath: Optional[str] = None, autoplay: bool = False) -> str:
        """
        Convert text to speech using Google Text-to-Speech (gTTS).
        
        Args:
            text (str): Text to convert to speech
            output_filepath (str, optional): Path to save the audio file
            autoplay (bool): Whether to automatically play the audio
            
        Returns:
            str: Path to the generated audio file
            
        Raises:
            APIError: If TTS generation fails
        """
        try:
            if not output_filepath:
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_filepath = temp_file.name
                temp_file.close()
            
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang='en',
                slow=False
            )
            
            # Save the audio file
            tts.save(output_filepath)
            logger.info(f"Audio saved to {output_filepath}")
            
            # Autoplay if requested
            if autoplay:
                self._play_audio(output_filepath)
            
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error in gTTS text-to-speech: {e}")
            raise APIError(f"Failed to generate speech with gTTS: {str(e)}")
    
    def text_to_speech_elevenlabs(self, text: str, output_filepath: Optional[str] = None, 
                                 voice: str = "Aria", autoplay: bool = False) -> str:
        """
        Convert text to speech using ElevenLabs API.
        
        Args:
            text (str): Text to convert to speech
            output_filepath (str, optional): Path to save the audio file
            voice (str): Voice to use for generation
            autoplay (bool): Whether to automatically play the audio
            
        Returns:
            str: Path to the generated audio file
            
        Raises:
            ConfigurationError: If ElevenLabs API key is not configured
            APIError: If TTS generation fails
        """
        if not self.elevenlabs_client:
            raise ConfigurationError("ElevenLabs API key not configured")
        
        try:
            if not output_filepath:
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_filepath = temp_file.name
                temp_file.close()
            
            # Generate audio
            audio = self.elevenlabs_client.generate(
                text=text,
                voice=voice,
                output_format="mp3_22050_32",
                model="eleven_turbo_v2"
            )
            
            # Save the audio file
            elevenlabs.save(audio, output_filepath)
            logger.info(f"ElevenLabs audio saved to {output_filepath}")
            
            # Autoplay if requested
            if autoplay:
                self._play_audio(output_filepath)
            
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error in ElevenLabs text-to-speech: {e}")
            raise APIError(f"Failed to generate speech with ElevenLabs: {str(e)}")
    
    def record_audio(self, output_filepath: Optional[str] = None, timeout: int = 20, 
                    phrase_time_limit: Optional[int] = None) -> str:
        """
        Record audio from the microphone and save as MP3.
        
        Args:
            output_filepath (str, optional): Path to save the recorded audio
            timeout (int): Maximum time to wait for speech to start
            phrase_time_limit (int, optional): Maximum time for the phrase
            
        Returns:
            str: Path to the recorded audio file
            
        Raises:
            APIError: If recording fails
        """
        try:
            if not output_filepath:
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_filepath = temp_file.name
                temp_file.close()
            
            recognizer = sr.Recognizer()
            
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                print("🎤 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("Start speaking now...")
                print("🎤 Start speaking now...")
                
                # Record the audio
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logger.info("Recording complete.")
                print("✅ Recording complete.")
                
                # Convert to MP3
                wav_data = audio_data.get_wav_data()
                audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                audio_segment.export(output_filepath, format="mp3", bitrate="128k")
                
                logger.info(f"Audio saved to {output_filepath}")
                return output_filepath
                
        except sr.WaitTimeoutError:
            logger.error("Recording timeout - no speech detected")
            raise APIError("Recording timeout - no speech detected")
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise APIError(f"Failed to record audio: {str(e)}")
    
    def speech_to_text_groq(self, audio_filepath: str) -> str:
        """
        Convert speech to text using Groq's Whisper API.
        
        Args:
            audio_filepath (str): Path to the audio file
            
        Returns:
            str: Transcribed text
            
        Raises:
            ConfigurationError: If Groq API key is not configured
            APIError: If transcription fails
        """
        if not self.groq_client:
            raise ConfigurationError("Groq API key not configured")
        
        try:
            with open(audio_filepath, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="en"
                )
            
            transcribed_text = transcription.text
            logger.info(f"Transcription successful: {transcribed_text[:50]}...")
            print(f"🎯 Transcribed: {transcribed_text}")
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error in Groq speech-to-text: {e}")
            raise APIError(f"Failed to transcribe audio with Groq: {str(e)}")
    
    def speech_to_text_google(self, audio_filepath: str) -> str:
        """
        Convert speech to text using Google Speech Recognition (offline fallback).
        
        Args:
            audio_filepath (str): Path to the audio file
            
        Returns:
            str: Transcribed text
            
        Raises:
            APIError: If transcription fails
        """
        try:
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_filepath) as source:
                audio_data = recognizer.record(source)
            
            # Recognize speech
            transcribed_text = recognizer.recognize_google(audio_data)
            logger.info(f"Google transcription successful: {transcribed_text[:50]}...")
            print(f"🎯 Transcribed: {transcribed_text}")
            
            return transcribed_text
            
        except sr.UnknownValueError:
            logger.error("Google Speech Recognition could not understand audio")
            raise APIError("Could not understand the audio")
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            raise APIError(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in Google speech-to-text: {e}")
            raise APIError(f"Failed to transcribe audio: {str(e)}")
    
    def _play_audio(self, audio_filepath: str) -> None:
        """
        Play audio file based on the operating system.
        
        Args:
            audio_filepath (str): Path to the audio file to play
        """
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', audio_filepath], check=True)
            elif os_name == "Windows":  # Windows
                subprocess.run(['powershell', '-c', 
                              f'(New-Object Media.SoundPlayer "{audio_filepath}").PlaySync();'], 
                              check=True)
            elif os_name == "Linux":  # Linux
                subprocess.run(['aplay', audio_filepath], check=True)
            else:
                logger.warning(f"Unsupported operating system for audio playback: {os_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error playing audio: {e}")
        except Exception as e:
            logger.error(f"Unexpected error playing audio: {e}")
    
    def process_voice_input(self, timeout: int = 20) -> Tuple[str, str]:
        """
        Complete voice input processing: record audio and convert to text.
        
        Args:
            timeout (int): Maximum time to wait for speech
            
        Returns:
            Tuple[str, str]: (audio_filepath, transcribed_text)
            
        Raises:
            APIError: If voice processing fails
        """
        try:
            # Record audio
            audio_filepath = self.record_audio(timeout=timeout)
            
            # Try Groq first, fallback to Google
            try:
                transcribed_text = self.speech_to_text_groq(audio_filepath)
            except (ConfigurationError, APIError) as e:
                logger.warning(f"Groq transcription failed, trying Google: {e}")
                transcribed_text = self.speech_to_text_google(audio_filepath)
            
            return audio_filepath, transcribed_text
            
        except Exception as e:
            logger.error(f"Error in voice input processing: {e}")
            raise APIError(f"Failed to process voice input: {str(e)}")
    
    def generate_voice_response(self, text: str, use_elevenlabs: bool = False, 
                              autoplay: bool = True) -> str:
        """
        Generate voice response from text.
        
        Args:
            text (str): Text to convert to speech
            use_elevenlabs (bool): Whether to use ElevenLabs (premium) or gTTS (free)
            autoplay (bool): Whether to automatically play the audio
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            if use_elevenlabs and self.elevenlabs_client:
                return self.text_to_speech_elevenlabs(text, autoplay=autoplay)
            else:
                return self.text_to_speech_gtts(text, autoplay=autoplay)
        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            # Fallback to gTTS if ElevenLabs fails
            if use_elevenlabs:
                logger.info("Falling back to gTTS")
                return self.text_to_speech_gtts(text, autoplay=autoplay)
            raise

# Global voice service instance
voice_service = VoiceService()

def get_voice_service() -> VoiceService:
    """Get the global voice service instance."""
    return voice_service
