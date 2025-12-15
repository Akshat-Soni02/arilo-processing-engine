from typing import Annotated
from fastapi import FastAPI, UploadFile, File
from services.audio_augmentation import AudioAugmentation
from services.llm_service import ConfigDict
from common.logging import get_logger

app = FastAPI()
logger = get_logger(__name__)

# TODO
# Auth middleware for inter service requests
# DB Setup for process state management & metadata storing


# Server health check endpoint
@app.get("/health")
def health():
    logger.info("Health check working")
    return {"status": "ok"}


# Testing endpoints


# TOREMOVE
# Add support to take audios as input  done
# log audio format & audio properties  already
# log data came after processing audio - processing time,
# processed audio time, processed audio format [if changed]
# Appropriate errors when - failed to find lib, failed to process due to upstream, failure due to server issues, warnings when taking longer then expected time [comparison against predetermined time with ref to some metric]
@app.post("/augment-audio")
async def augment_audio(audio_file: Annotated[UploadFile, File()]):
    audio_bytes = await audio_file.read()

    service = AudioAugmentation({})
    audio = service.run_pipeline(audio_bytes)

    with open("output_audio.wav", "wb") as f:
        f.write(audio)
    return {"message": "Audio augmentation endpoint"}


@app.post("/llm-test")
def llm_test():
    from services.llm_service import LLMService

    config = {
        "provider": "gemini",
        "api_key": "AQ.Ab8RN6LbX-AmjvGNCW_E0Q7gi30mD7atLhLcGCHEKJbHQniZbw",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.5,
        "max_tokens": 150,
    }

    config = ConfigDict(**config)
    llm_service = LLMService(config)
    response = llm_service.process("Hello, how are you?")
    return {"llm_response": response.content}
