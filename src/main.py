from typing import Annotated
from fastapi import FastAPI, UploadFile, File
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from services.pubsub.pubsub_service import PubSubService
from common.logging import get_logger, configure_logging
from config.settings import ARILO_SUBSCRIPTION_ID, APP_ENV, LOG_LEVEL

# Configure logging
configure_logging(env=APP_ENV, level=LOG_LEVEL)
logger = get_logger(__name__)

app = FastAPI(title="Arilo Processing Engine", version="1.0.0")

# Global state
listener_future = None
pubsub_service = PubSubService(SUBSCRIPTION_ID=ARILO_SUBSCRIPTION_ID)
@app.on_event("startup")
async def startup_event():
    """Starts the Pub/Sub listener on application startup."""
    global listener_future
    logger.info("Starting Pub/Sub listener...")

    def run_listener():
        global listener_future
        listener_future = pubsub_service.start_listener()
        try:
            listener_future.result()
        except Exception as e:
            print(f"Listener error: {e}")
    listener_thread = threading.Thread(target=run_listener, daemon=True)
    listener_thread.start()
    logger.info("Pub/Sub listener started.")
    
@app.on_event("shutdown")
async def shutdown_event():
    """Stops the Pub/Sub listener on application shutdown."""
    global listener_future
    if listener_future:
        logger.info("Stopping Pub/Sub listener...")
        pubsub_service.stop_listener(listener_future)
        logger.info("Pub/Sub listener stopped.")

@app.post("/publish")
async def publish_message(data: dict, attributes: Dict[str, str] | None = None):
    """Publish a message to Pub/Sub with optional attributes."""
    try:
        message_id = pubsub_service.publish_message(data, attributes=attributes or {})
        return {"status": "success", "message_id": message_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Server health check endpoint
@app.get("/health")
def health():
    logger.info("Health check working")
    return {"status": "ok"}


