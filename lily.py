import os
import json
import random
import requests
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
# Hugging Face API
API_TOKEN = os.environ.get("HF_API_TOKEN") # Securely load from environment variable
print(f"DEBUG: API_TOKEN retrieved is: '{API_TOKEN}'")
Headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Hugging Face Model Endpoints
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_API_URL =  "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
GENERATION_API_URL = "https://api-inference.huggingface.co/models/gpt2"

# --- Data Storage ---
ACHIEVEMENTS_FILE = 'achievements.json'
STORE_FILE = 'store.json'

def load_data(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    with open(filename, 'r') as f:
        return json.load(f)

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

achievements = load_data(ACHIEVEMENTS_FILE)
store = load_data(STORE_FILE)

# --- Hugging Face API Functions ---
def query(api_url, payload):
    if not API_TOKEN:
        print("Error: Hugging Face API Token (HF_API_TOKEN) not found. Please set the environment variable.")
        return None
    try:
        response = requests.post(api_url, headers=Headers, json=payload)
        response.raise_for_status() # This will raise an HTTPError for 4xx/5xx responses (like 404)
        return response.json()
    except requests.exceptions.RequestException as e:
        # Explicitly print the error and the URL that failed
        print(f"API request failed: {e} for url: {api_url}") # Changed line
        return None # Ensure None is returned on ANY request exception
    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON, but no HTTP error
        print(f"API response was not valid JSON for url: {api_url}")
        return None

def get_sentiment(text):
    output = query(SENTIMENT_API_URL, {"inputs": text})
    if output and isinstance(output, list) and output:
        # Expected format: [[{'label': 'POSITIVE', 'score': 0.99}, {'label': 'NEGATIVE', 'score': 0.00}]]
        # We take the first item, then the item with the highest score
        best_sentiment = max(output[0], key=lambda x: x['score'])
        return best_sentiment['label']
    return "NEUTRAL"

def classify_zero_shot(text, candidate_labels):
    output = query(ZERO_SHOT_API_URL, {"inputs": text, "parameters": {"candidate_labels": candidate_labels}})
    if output and 'labels' in output and 'scores' in output:
        # Returns the label with the highest score
        return output['labels'][0]
    return "UNKNOWN"

def generate_text(prompt, max_length=100, num_return_sequences=1):
    output = query(GENERATION_API_URL, {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "num_return_sequences": num_return_sequences}
    })
    if output and isinstance(output, list) and output and 'generated_text' in output[0]:
        return output[0]['generated_text']
    return "I'm having trouble generating text right now."

# --- Core Logic Functions ---
def handle_achievement(user_input):
    global achievements
    classification = classify_zero_shot(user_input, ["achievement", "goal", "success", "milestone", "win", "progress","not_an_achievement"])
    if classification == "achievement" or classification == "success" or classification == "milestone" or classification == "win":
        achievements.append(user_input)
        save_data(achievements, ACHIEVEMENTS_FILE)
        print("That's fantastic! I've added that to your achievements. Keep up the great work!")
        return True
    return False

def get_encouragement():
    if achievements:
        recent_achievement = random.choice(achievements)
        return f"Remember when you achieved '{recent_achievement}'? You're capable of amazing things, keep pushing forward!"
    else:
        quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Believe you can and you're halfway there. - Theodore Roosevelt",
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt"
        ]
        return random.choice(quotes)

def get_story_or_quote(mood):
    if mood == "NEGATIVE":
        return generate_text("Tell me a short motivational story about overcoming challenges:")  
    return "Sometimes, just a listening ear is enough. I'm here."


def get_random_positive_moment():
    if store:
        return random.choice(store)
    return "Keep noting down your positive moments, they add up!"

# --- Main Interaction Loop ---
def lily_chat():
    print("Hello! I'm Lily, your emotional companion. May I know your good name?")
    name = input("")
    while True:
        user_input = input(f"Hi {name},How are you feeling today?").strip()
        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Goodbye! Remember, I'm here whenever you need me.")
            break

        sentiment = get_sentiment(user_input)

        if sentiment == "POSITIVE":
            print("That sounds wonderful! I'm happy for you.Wanna share what made you so happy?")
            moment = input("")
            if handle_achievement(user_input):
                continue # If it's an achievement, just confirm and continue
            print("Would you like to save this positive moment? (yes/no)")
            savechoice = input(" ").strip().lower()
            if savechoice == 'yes':
                store.append(user_input)
                save_data(store, STORE_FILE)
                print("Moment saved! We're building a collection of your happy times.")
                follow_up_prompt = f"The user said '{user_input}'. Lily is curious and wants to ask a follow-up question. Lily: "
                generated_follow_up = generate_text(follow_up_prompt, max_length=100) # Increased max_length slightly
                
                if generated_follow_up and generated_follow_up.startswith(follow_up_prompt) and len(generated_follow_up) > len(follow_up_prompt):
                    clean_follow_up = generated_follow_up[len(follow_up_prompt):].strip()
                    # Try to find the first question mark and use that as the end
                    if '?' in clean_follow_up:
                        clean_follow_up = clean_follow_up.split('?', 1)[0].strip() + '?'
                    elif '.' in clean_follow_up: # Or a period, and turn it into a question
                        clean_follow_up = clean_follow_up.split('.', 1)[0].strip() + '?'
                    
                    print(f"{clean_follow_up}")
                else:
                    print("Tell me more about that.") # Fallback
                      
        elif sentiment == "NEGATIVE":
            print("I hear that you're feeling down. I'm here for you.")
            print("Would you like some encouragement, a motivational story, or just a listening ear? (encouragement/story/listening)")
            support_choice = input("").strip().lower()
            if support_choice == "encouragement":
                print(f"{get_encouragement()}")
            elif support_choice == "story":
                story = get_story_or_quote("NEGATIVE")
                print(f"{story}")
            else:
                print("I'm here to listen. Take your time.")
            follow_up_prompt = f"The user said '{user_input}'. Lily is curious and wants to ask a follow-up question. Lily: "
            generated_follow_up = generate_text(follow_up_prompt, max_length=100) # Increased max_length slightly
                
            if generated_follow_up and generated_follow_up.startswith(follow_up_prompt) and len(generated_follow_up) > len(follow_up_prompt):
                clean_follow_up = generated_follow_up[len(follow_up_prompt):].strip()
                # Try to find the first question mark and use that as the end
               if '?' in clean_follow_up:
                    clean_follow_up = clean_follow_up.split('?', 1)[0].strip() + '?'
                elif '.' in clean_follow_up: # Or a period, and turn it into a question
                    clean_follow_up = clean_follow_up.split('.', 1)[0].strip() + '?'
                    
                    print(f"{clean_follow_up}")
                else:
                    print("Tell me more about that.") # Fallback
        else: # NEUTRAL or UNKNOWN sentiment
            print("Hmm, I'm not quite sure how you're feeling. Tell me more!")
            print("Perhaps you'd like to recall a past achievement or a positive moment? (achievement/moment/no)")
            recall_choice = input("").strip().lower()
            if recall_choice == "achievement":
                print(f"{get_encouragement()}")
            elif recall_choice == "moment":
                print(f"Here's a positive moment you shared: {get_random_positive_moment()}")
            else:
                print("Okay, I'm here if you want to talk about anything.")
             follow_up_prompt = f"The user said '{user_input}'. Lily is curious and wants to ask a follow-up question. Lily: "
                generated_follow_up = generate_text(follow_up_prompt, max_length=100) # Increased max_length slightly
                
                if generated_follow_up and generated_follow_up.startswith(follow_up_prompt) and len(generated_follow_up) > len(follow_up_prompt):
                    clean_follow_up = generated_follow_up[len(follow_up_prompt):].strip()
                    # Try to find the first question mark and use that as the end
                    if '?' in clean_follow_up:
                        clean_follow_up = clean_follow_up.split('?', 1)[0].strip() + '?'
                    elif '.' in clean_follow_up: # Or a period, and turn it into a question
                        clean_follow_up = clean_follow_up.split('.', 1)[0].strip() + '?'
                    
                    print(f"{clean_follow_up__}")
                else:
                    print("Tell me more about that.") # Fallback
if __name__ == "__main__":
    lily_chat()