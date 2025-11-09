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
test_dir = os.path.dirname(os.path.abspath(__file__))
# Path to your test audio file
audio_file_path = os.path.join(test_dir, "test_audio.wav")

try:
    with open(audio_file_path, "rb") as audio_file:
        print("🔄 Testing Whisper API connectivity...")
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )

        print("\n✅ Whisper API Connection Successful!")
        print("📝 Transcription result:")
        print(response.text)

except Exception as e:
    print("\n❌ Whisper API test failed.")
    print("Error details:")
    print(e)
