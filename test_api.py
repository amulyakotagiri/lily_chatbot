import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.environ.get("HF_API_TOKEN")
print(f"DEBUG: API_TOKEN in test_api.py is: '{API_TOKEN}'")

SENTIMENT_API_URL_TEST = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_API_URL_TEST = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli" # <-- ADD THIS LINE

Headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Your query function (ensure it's the latest improved version)
def query(api_url, payload):
    if not API_TOKEN:
        print("Error: Hugging Face API Token (HF_API_TOKEN) not found. Please set the environment variable.")
        return None
    try:
        response = requests.post(api_url, headers=Headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e} for url: {api_url}")
        return None
    except json.JSONDecodeError:
        print(f"API response was not valid JSON for url: {api_url}")
        return None

# Function to test the Sentiment API call
def test_sentiment_api(text):
    print(f"\nTesting sentiment for: '{text}' using URL: {SENTIMENT_API_URL_TEST}")
    test_output = query(SENTIMENT_API_URL_TEST, {"inputs": text})
    if test_output:
        print(f"Test API SUCCESS: {test_output}")
    else:
        print("Test API FAILED to get a result.")
    return test_output

# <-- ADD THIS NEW FUNCTION BELOW test_sentiment_api -->
def test_zero_shot_api(text, candidate_labels):
    print(f"\nTesting zero-shot classification for: '{text}' with labels: {candidate_labels} using URL: {ZERO_SHOT_API_URL_TEST}")
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    test_output = query(ZERO_SHOT_API_URL_TEST, payload)
    if test_output:
        print(f"Test API SUCCESS: {test_output}")
    else:
        print("Test API FAILED to get a result.")
    return test_output
# <-- END NEW FUNCTION -->


# Run the tests
print("\n--- Running API Tests ---")
test_result_sentiment = test_sentiment_api("I am feeling great today!")
test_result_zero_shot_good = test_zero_shot_api("good", ["achievement", "goal", "success", "milestone", "win", "progress", "general feeling"]) # Add "general feeling" to see if it prefers that
test_result_zero_shot_achievement = test_zero_shot_api("I finished my project!", ["achievement", "goal", "success"]) # Test a real achievement