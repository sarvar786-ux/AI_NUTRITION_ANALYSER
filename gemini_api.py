import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Load model
model = genai.GenerativeModel("gemini-1.5-flash")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def analyze_food_image(image_path):
    """
    Analyze food image using Gemini Vision
    """

    try:
        image = Image.open(image_path)

        response = model.generate_content([
            "List all food items in this image separated by commas.",
            image
        ])

        logging.debug(f"Gemini Response: {response.text}")

        if not response.text.strip():
            return []

        food_items = [
            item.strip()
            for item in response.text.split(",")
            if item.strip()
        ]

        return food_items

    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        return []


def generate_health_summary(summary):

    try:
        prompt = f"""
        Based on this nutrition data, give short health advice and improvement tips:

        {summary}

        Keep response simple and 3-4 bullet points.
        """

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "Unable to generate AI health summary."
        
