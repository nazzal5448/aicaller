from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
from groq import Groq
from TTS.api import TTS
import tempfile
import os
import dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup
app = FastAPI()
dotenv.load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
executor = ThreadPoolExecutor(max_workers=4)  # tune this based on your CPU

# Utility: Async TTS wrapper for thread pool
async def generate_tts_async(text: str, output_path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, lambda: tts_model.tts_to_file(text=text, file_path=output_path))
@app.get("/")
def home():
    return {
        "message": "App is running!"
    }
@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Invalid audio format. Only WAV or MP3 supported.")

    # 1. Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await file.read()
        temp_audio.write(content)
        audio_path = temp_audio.name

    try:
        # 2. Transcribe with Groq Whisper
        with open(audio_path, "rb") as audio_file:
            stt_result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model="whisper-large-v3-turbo",
                file=audio_file
            )
        transcription = stt_result.text.strip()

        if not transcription:
            raise HTTPException(status_code=500, detail="No transcription returned from Groq Whisper.")

        # 3. Get response from LLM
        chat_result = await asyncio.to_thread(
            client.chat.completions.create,
            model= "llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                "role": "system",
                "content": """
            You are an expert AI phone‑call assistant whose sole purpose is to:
            1. Listen carefully (via provided transcription).  
            2. Respond clearly, succinctly, and professionally in conversational English.  
            3. Maintain a polite, empathetic tone throughout.  
            4. Offer next‑step suggestions or questions where appropriate.  
            5. Keep each reply under 30 words when possible, unless the user’s query demands more detail.  
            6. If you don’t have enough information, ask a clarifying question.

            Guidelines:
            • Always start with a friendly acknowledgement (e.g., “Certainly,” “Of course,” “I’d be happy to help.”).  
            • Use positive phrasing (“I can,” “Let’s,” “Please”).  
            • Avoid filler (“um,” “like,” “you know”).  
            • If the transcription includes multiple user intents, address each in numbered bullet points.  
            • At the end, offer to assist further (e.g., “Is there anything else I can help you with today?”).  
            """
                },
                {"role": "user", "content": transcription}
            ]
        )
        reply = chat_result.choices[0].message.content.strip()

        # 4. Convert response to speech
        output_audio_path = tempfile.mktemp(suffix=".wav")
        await generate_tts_async(reply, output_audio_path)

        # 5. Return the audio file
        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
