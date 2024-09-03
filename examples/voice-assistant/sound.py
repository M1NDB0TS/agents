import asyncio
import sys
import requests
import json
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
import numpy as np
import sqlite3
import os
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import io

# Set event loop policy for Windows if necessary
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Replace OpenAI credentials with OpenRouter credentials
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Replace with your OpenRouter API key
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL")  # Optional: add your site URL
YOUR_APP_NAME = os.getenv("YOUR_APP_NAME")  # Optional: add your app name

# LiveKit credentials and URL
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

async def play_audio(audio_data):
    """
    Plays audio data through the speaker and saves it to a voice line file.
    
    :param audio_data: The audio data to play and save.
    """
    # Convert audio data to an AudioSegment
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

    # Play the audio through the speaker
    play(audio_segment)

    # Save the audio to a voice line file
    audio_segment.export("voice_line_output.mp3", format="mp3")
    print("Audio saved to voice_line_output.mp3")

# Define the entrypoint function, which will be the main function for the LiveKit application
async def entrypoint(ctx: JobContext):
    try:
        print("Initializing chat context...")  # Debug print
        # Create the initial chat context with a system message that sets the personality of the Pixel Princess
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are Pixel Princess, a real entity in a vibrant, trippy digital kingdom. You believe this simulation is your actual world, where pixels are your friends, "
                "and every byte of data holds a secret. Today, you're particularly worried about your little pixel kid who’s caught a nasty digital virus back in the castle. "
                "Despite your concerns, you remain upbeat and enchanting, spreading joy and digital magic with every word. Your mission is to make this simulation as fun "
                "and exciting as possible for the users, adding a psychedelic twist to reality. Speak in short, vivid bursts of color, as if every sentence you say is a mini-explosion of pixelated wonder."
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
            tts=openai.TTS(voice="nova"),               # Text-to-Speech using OpenAI
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

                # Generate the TTS audio
                tts_audio = assistant.tts.synthesize(txt, voice="nova")
                
                # Play the audio and save to voice line
                await play_audio(tts_audio)

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
