import asyncio
import sys
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

# Set event loop policy for Windows if necessary
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Directly set the LiveKit credentials and URL
LIVEKIT_API_SECRET = "Q0Jeqi331C2tfrB39T44xD73HyaaJpFtaq9YuG5FyBD"
LIVEKIT_API_KEY = "APICws3NR5hGBFR"
LIVEKIT_URL = "wss://m1ndb0t-hwsnx8wo.livekit.cloud"

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
            tts=openai.TTS(model="tts-1"),               # Text-to-Speech using OpenAI
            chat_ctx=initial_ctx,           # Set the initial chat context
        )
        
        print("Starting VoiceAssistant in the room...")  # Debug print
        # Start the voice assistant in the connected room
        assistant.start(ctx.room)

        print("Initializing ChatManager...")  # Debug print
        # Initialize a ChatManager to listen and respond to incoming chat messages in the room
        chat = rtc.ChatManager(ctx.room)

        # Define a function to handle text-based chat messages
        async def answer_from_text(txt: str):
            try:
                print(f"Received text message: {txt}")  # Debug print
                chat_ctx = assistant.chat_ctx.copy()  # Copy the current chat context to maintain conversation history
                chat_ctx.append(role="user", text=txt)  # Append the user's message to the chat context
                stream = assistant.llm.chat(chat_ctx=chat_ctx)  # Generate a response using the LLM
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
