from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import pickle
from typing import Dict, Optional
import uvicorn
import redis
from yandex_cloud_ml_sdk import YCloudML

# Define request and response models
class UserRequest(BaseModel):
    message: str
    chat_id: str

class ModelResponse(BaseModel):
    response: str

# Initialize FastAPI app
app = FastAPI(title="AI Assistant API", 
              description="API for processing user messages with a language model",
              version="1.0.0")

# Dictionary to store chat threads (in-memory cache)
threads = {}

# Initialize Redis connection
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))
redis_password = os.environ.get("REDIS_PASSWORD", None)
redis_client = None

try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False  # We need binary responses for pickle
    )
    # Test the connection
    redis_client.ping()
    print("Redis connection established successfully")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    redis_client = None

# Functions for thread persistence
def save_thread_to_redis(chat_id, thread):
    """Save a thread to Redis"""
    if redis_client:
        try:
            # Serialize the thread object
            thread_data = pickle.dumps(thread)
            # Save to Redis with a key pattern 'thread:{chat_id}'
            redis_client.set(f"thread:{chat_id}", thread_data)
            return True
        except Exception as e:
            print(f"Error saving thread to Redis: {e}")
    return False

def load_thread_from_redis(chat_id):
    """Load a thread from Redis"""
    if redis_client:
        try:
            # Get the serialized thread data from Redis
            thread_data = redis_client.get(f"thread:{chat_id}")
            if thread_data:
                # Deserialize the thread object
                thread = pickle.loads(thread_data)
                return thread
        except Exception as e:
            print(f"Error loading thread from Redis: {e}")
    return None

# Initialize the YandexGPT model
# In a production environment, these would be set as environment variables
try:
    folder_id = os.environ.get("FOLDER_ID", "your_folder_id")
    api_key = os.environ.get("API_KEY", "your_api_key")

    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    model = sdk.models.completions("yandexgpt", model_version="latest")
except Exception as e:
    print(f"Error initializing model: {e}")
    # For demo purposes, we'll continue without a real model
    sdk = None
    model = None

# Function to create a new thread
def create_thread():
    if sdk:
        return sdk.threads.create(ttl_days=1, expiration_policy="static")
    return None

# Function to create an assistant
def create_assistant(model, tools=None):
    if not sdk or not model:
        return None

    kwargs = {}
    if tools and len(tools) > 0:
        kwargs = {"tools": tools}

    assistant = sdk.assistants.create(
        model, ttl_days=1, expiration_policy="since_last_active", **kwargs
    )

    assistant.update(
        instruction="""Ты - опытный ассистент, задача которого - помогать пользователю 
        отвечать на вопросы и выполнять задачи."""
    )

    return assistant

# Create a global assistant
assistant = create_assistant(model) if sdk and model else None

# Get or create a thread for a chat_id
def get_thread(chat_id):
    # First check in-memory cache
    if chat_id in threads:
        return threads[chat_id]

    # Then check Redis
    thread = load_thread_from_redis(chat_id)
    if thread:
        # Update in-memory cache
        threads[chat_id] = thread
        return thread

    # Create new thread if not found
    if sdk:
        thread = create_thread()
        threads[chat_id] = thread
        # Save to Redis for persistence
        save_thread_to_redis(chat_id, thread)
        return thread

    return None

# Process message with model
def process_message(message, chat_id):
    if not sdk or not model or not assistant:
        # For demo purposes, return a mock response
        return f"Mock response to: {message}"

    thread = get_thread(chat_id)
    if not thread:
        raise HTTPException(status_code=500, detail="Failed to create thread")

    thread.write(message)
    run = assistant.run(thread)
    result = run.wait()

    return result.text

@app.post("/process", response_model=ModelResponse)
async def process_request(request: UserRequest):
    try:
        response = process_message(request.message, request.chat_id)
        return ModelResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the server when the script is executed directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
