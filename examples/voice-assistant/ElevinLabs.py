import asyncio
import sys
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
import numpy as np
import sqlite3
import json
import elevenlabs


# Set event loop policy for Windows if necessary
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Directly set the LiveKit credentials and URL
LIVEKIT_API_SECRET = "Q0Jeqi331C2tfrB39T44xD73HyaaJpFtaq9YuG5FyBD"
LIVEKIT_API_KEY = "APICws3NR5hGBFR"
LIVEKIT_URL = "wss://m1ndb0t-hwsnx8wo.livekit.cloud"

# Database connection setup (SQLite for simplicity)
DB_PATH = "embeddings.db"  # Define the path for your SQLite database

def create_db():
    """
    Creates a SQLite database to store embeddings if it doesn't already exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def save_embedding_to_db(text, embedding):
    """
    Saves the text and its corresponding embedding vector to the database.
    
    :param text: The input text string.
    :param embedding: The corresponding embedding vector as a list of floats.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)", 
                   (text, json.dumps(embedding)))
    conn.commit()
    conn.close()

def search_embedding_in_db(embedding, threshold=0.8):
    """
    Searches for similar embeddings in the database.
    
    :param embedding: The embedding vector to search for as a list of floats.
    :param threshold: The cosine similarity threshold to consider a match.
    :return: List of similar texts from the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text, embedding FROM embeddings")
    results = cursor.fetchall()
    conn.close()

    similar_texts = []
    for result in results:
        stored_embedding = json.loads(result[1])
        similarity = cosine_similarity(np.array(stored_embedding), np.array(embedding))
        if similarity > threshold:
            similar_texts.append(result[0])

    return similar_texts

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    
    :param vec1: First vector as a numpy array.
    :param vec2: Second vector as a numpy array.
    :return: Cosine similarity score between the two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Define the entrypoint function, which will be the main function for the LiveKit application
async def entrypoint(ctx: JobContext):
    try:
        print("Initializing chat context...")  # Debug print
        # Create the initial chat context with a system message that sets the personality of the voice assistant
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
                "You should use short and concise responses, and avoid usage of unpronounceable punctuation."
            ),
        )

        print("Connecting to LiveKit room...")  # Debug print
        # Connect to the LiveKit room with audio-only auto-subscribe settings
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        print("Initializing VoiceAssistant...")  # Debug print
        # Initialize the VoiceAssistant with various plugins for voice activity detection, speech-to-text, 
        # large language model (LLM) chat, and text-to-speech functionalities.
        assistant = VoiceAssistant(
            vad=silero.VAD.load(),          # Voice Activity Detection using Silero
            stt=deepgram.STT(),             # Speech-to-Text using Deepgram
            llm=openai.LLM(model="gpt-4o-mini"),               # Language Model using OpenAI
            tts = elevenlabs.TTS(model_id="eleven_turbo_v2"),
            chat_ctx=initial_ctx,           # Set the initial chat context
        )
        
        print("Starting VoiceAssistant in the room...")  # Debug print
        # Start the voice assistant in the connected room
        assistant.start(ctx.room)

        print("Initializing ChatManager...")  # Debug print
        # Initialize a ChatManager to listen and respond to incoming chat messages in the room
        chat = rtc.ChatManager(ctx.room)

        # Create the database for storing embeddings if it doesn't exist
        create_db()

        # Define a function to handle text-based chat messages
        async def answer_from_text(txt: str):
            try:
                print(f"Received text message: {txt}")  # Debug print
                chat_ctx = assistant.chat_ctx.copy()  # Copy the current chat context to maintain conversation history
                chat_ctx.append(role="user", text=txt)  # Append the user's message to the chat context
                stream = assistant.llm.chat(chat_ctx=chat_ctx)  # Generate a response using the LLM

                # Generate the embedding for the input text
                embedding = openai.Embeddings.create(
                    input=txt,
                    model="text-embedding-3-small"
                ).data[0].embedding

                print(f"Embedding generated for text: {embedding}")  # Debug print

                # Save the text and its embedding to the database
                save_embedding_to_db(txt, embedding)
                print("Embedding saved to database.")  # Debug print

                # Check if there are similar embeddings in the database
                similar_texts = search_embedding_in_db(embedding)
                if similar_texts:
                    print("Similar texts found:", similar_texts)  # Debug print
                    await assistant.say(f"I've found similar texts: {', '.join(similar_texts)}", allow_interruptions=True)

                await assistant.say(stream)  # Convert the response to speech and play it in the room
            except Exception as e:
                print(f"Error processing text message: {e}")  # Error log

        # Listen for incoming chat messages and respond using the assistant
        @chat.on("message_received")
        def on_chat_received(msg: rtc.ChatMessage):
            if msg.message:
                print(f"Chat message received: {msg.message}")  # Debug print
                asyncio.create_task(answer_from_text(msg.message))  # Process and respond to the message asynchronously

        # Initial greeting message to the user
        await asyncio.sleep(1)  # Short delay before speaking
        await assistant.say("Hey, how can I help you today?", allow_interruptions=True)  # Greet the user
        print("Greeting message sent.")  # Debug print

    except Exception as e:
        print(f"Error in entrypoint: {e}")  # Error log

# Main entry point for the application
if __name__ == "__main__":
    try:
        print("Starting application...")  # Debug print
        # Run the application with the specified worker options, setting the entrypoint function
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        print(f"Error in main application: {e}")  # Error log
