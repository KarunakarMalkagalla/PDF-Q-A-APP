# Import new Gemini SDK
from google import genai

# Import dotenv loader
from dotenv import load_dotenv

# Access environment variables
import os

# Load variables from .env
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Send prompt to Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say exactly: Hello from Gemini!"
)

# Print response
print(response.text)