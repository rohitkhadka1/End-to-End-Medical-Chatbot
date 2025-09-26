#!/usr/bin/env python3
"""
Voice Features Test Script for Mental Health Chatbot
This script helps you test the voice functionality independently.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def test_text_to_speech():
    """Test text-to-speech functionality."""
    print("=== Testing Text-to-Speech ===")
    
    try:
        from voice_service import get_voice_service
        voice_service = get_voice_service()
        
        test_text = "Hello! This is a test of the text-to-speech functionality for the mental health chatbot."
        
        print(f"Converting text to speech: {test_text}")
        
        # Test gTTS (free)
        print("\n1. Testing gTTS (Google Text-to-Speech)...")
        try:
            audio_path = voice_service.text_to_speech_gtts(test_text, autoplay=True)
            print(f"✅ gTTS successful! Audio saved to: {audio_path}")
        except Exception as e:
            print(f"❌ gTTS failed: {e}")
        
        # Test ElevenLabs (premium) if API key is available
        if voice_service.elevenlabs_client:
            print("\n2. Testing ElevenLabs (Premium Text-to-Speech)...")
            try:
                audio_path = voice_service.text_to_speech_elevenlabs(test_text, autoplay=True)
                print(f"✅ ElevenLabs successful! Audio saved to: {audio_path}")
            except Exception as e:
                print(f"❌ ElevenLabs failed: {e}")
        else:
            print("\n2. ElevenLabs API key not configured - skipping premium TTS test")
            
    except Exception as e:
        print(f"❌ Text-to-speech test failed: {e}")

def test_speech_to_text():
    """Test speech-to-text functionality."""
    print("\n=== Testing Speech-to-Text ===")
    
    try:
        from voice_service import get_voice_service
        voice_service = get_voice_service()
        
        print("This will record audio from your microphone for 10 seconds...")
        input("Press Enter when you're ready to start recording...")
        
        # Record audio
        print("🎤 Recording... Speak now!")
        audio_path = voice_service.record_audio(timeout=10)
        print(f"✅ Recording saved to: {audio_path}")
        
        # Test Groq transcription
        if voice_service.groq_client:
            print("\n1. Testing Groq (Whisper) Speech-to-Text...")
            try:
                transcription = voice_service.speech_to_text_groq(audio_path)
                print(f"✅ Groq transcription: '{transcription}'")
            except Exception as e:
                print(f"❌ Groq transcription failed: {e}")
        else:
            print("\n1. Groq API key not configured - skipping Groq STT test")
        
        # Test Google Speech Recognition (fallback)
        print("\n2. Testing Google Speech Recognition (fallback)...")
        try:
            transcription = voice_service.speech_to_text_google(audio_path)
            print(f"✅ Google transcription: '{transcription}'")
        except Exception as e:
            print(f"❌ Google transcription failed: {e}")
            
    except Exception as e:
        print(f"❌ Speech-to-text test failed: {e}")

def test_complete_voice_interaction():
    """Test complete voice interaction."""
    print("\n=== Testing Complete Voice Interaction ===")
    
    try:
        from voice_service import get_voice_service
        voice_service = get_voice_service()
        
        print("This will test the complete voice interaction:")
        print("1. Record your voice")
        print("2. Convert speech to text")
        print("3. Generate a response")
        print("4. Convert response to speech")
        
        input("Press Enter when you're ready...")
        
        # Step 1 & 2: Record and transcribe
        audio_path, user_input = voice_service.process_voice_input(timeout=15)
        print(f"🎯 You said: '{user_input}'")
        
        # Step 3: Generate a simple response (without the full chatbot pipeline)
        response_text = f"Thank you for saying: '{user_input}'. This is a test response from the voice system."
        
        # Step 4: Convert response to speech
        print("🔊 Converting response to speech...")
        response_audio = voice_service.generate_voice_response(response_text, autoplay=True)
        print(f"✅ Complete voice interaction successful!")
        print(f"Input audio: {audio_path}")
        print(f"Response audio: {response_audio}")
        
    except Exception as e:
        print(f"❌ Complete voice interaction test failed: {e}")

def main():
    """Main test function."""
    print("🎙️ Mental Health Chatbot - Voice Features Test")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['HUGGINGFACEHUB_API_TOKEN', 'PINECONE_API_KEY']
    optional_vars = ['ELEVEN_API_KEY', 'GROQ_API_KEY']
    
    print("Environment Check:")
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Configured")
        else:
            print(f"❌ {var}: Missing (required)")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var}: Configured")
        else:
            print(f"⚠️ {var}: Missing (optional - some features won't work)")
    
    print("\nAvailable Tests:")
    print("1. Text-to-Speech Test")
    print("2. Speech-to-Text Test")
    print("3. Complete Voice Interaction Test")
    print("4. Run All Tests")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                test_text_to_speech()
            elif choice == '2':
                test_speech_to_text()
            elif choice == '3':
                test_complete_voice_interaction()
            elif choice == '4':
                test_text_to_speech()
                test_speech_to_text()
                test_complete_voice_interaction()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()
