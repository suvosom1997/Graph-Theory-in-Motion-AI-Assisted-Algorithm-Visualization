import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Get the directory where the test file is located
speech_file_path = "output_speech.mp3"

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",  # or "tts-1"
    voice="alloy",            # other options: verse, coral, etc.
    input="Hello! This is a test of OpenAI's text-to-speech model."
)

# Save audio to file
response.stream_to_file(speech_file_path)
print("✅ Text-to-speech file saved as", speech_file_path)