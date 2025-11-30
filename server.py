"""
ChatterBox Multilingual TTS Server
Hosts the ChatterBox multilingual TTS model for audio generation via HTTP API.
Supports 23 languages with configurable exaggeration and CFG weight parameters.
"""
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import os
import logging
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model = None
model_sample_rate = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management"""
    # Startup
    initialize_model()
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(title="ChatterBox TTS API", lifespan=lifespan)

def initialize_model():
    """Initialize ChatterBox model on startup"""
    global model, model_sample_rate

    logger.info("Initializing ChatterBox multilingual TTS model...")

    # Detect device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Patch torch.load to use the correct device
    map_location = torch.device(device)
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    torch.load = patched_torch_load

    try:
        # Load Multilingual model (supports all 23 languages including English)
        logger.info("Loading ChatterboxMultilingualTTS model...")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        model_sample_rate = model.sr
        logger.info(f"ChatterBox model loaded successfully. Sample rate: {model_sample_rate}")

    except Exception as e:
        logger.error(f"Failed to initialize ChatterBox model: {e}", exc_info=True)
        raise

# Define request model
class GenerateAudioRequest(BaseModel):
    """Request schema for audio generation"""
    text: str                          # Text to generate speech for
    reference_audio_path: str          # Path to reference audio for voice cloning
    output_path: str                   # Path to save generated audio
    language_id: str = "en"            # Target language (2-letter ISO code)
    exaggeration: float = 0.5          # Emotion/intensity control (0.3-2.0)
    cfg_weight: float = 0.5            # Text adherence vs creativity (0.3-1.0)

@app.post("/generate-audio")
async def generate_audio(request: GenerateAudioRequest):
    """
    Generate audio using ChatterBox multilingual TTS.

    Args:
        text: Text to generate speech for
        reference_audio_path: Path to reference audio for voice cloning
        output_path: Path to save generated audio
        language_id: Target language (2-letter ISO code)
        exaggeration: Emotion control (0.3-2.0, default 0.5)
        cfg_weight: Text adherence control (0.3-1.0, default 0.5)

    Returns:
        JSON with status, output_path, sample_rate, and duration
    """
    try:
        logger.info("Received generate-audio request")
        logger.info(f"Text length: {len(request.text)} chars")
        logger.info(f"Reference audio: {request.reference_audio_path}")
        logger.info(f"Target language: {request.language_id}")
        logger.info(f"Exaggeration: {request.exaggeration}, CFG Weight: {request.cfg_weight}")

        # Validate inputs
        if not os.path.exists(request.reference_audio_path):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {request.reference_audio_path}"
            )

        # Validate parameter ranges
        if not (0.3 <= request.exaggeration <= 2.0):
            raise HTTPException(
                status_code=400,
                detail=f"Exaggeration must be between 0.3 and 2.0, got {request.exaggeration}"
            )

        if not (0.3 <= request.cfg_weight <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"CFG weight must be between 0.3 and 1.0, got {request.cfg_weight}"
            )

        # Ensure output directory exists
        output_dir = os.path.dirname(request.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("Starting ChatterBox generation...")
        logger.info(f"  Generate text: '{request.text[:100]}...'")
        logger.info(f"  Language: {request.language_id}")

        # Generate audio using ChatterBox
        wav = model.generate(
            text=request.text,
            audio_prompt_path=request.reference_audio_path,
            language_id=request.language_id,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight
        )

        # Save the generated audio
        ta.save(request.output_path, wav, model_sample_rate)

        # Calculate duration
        duration = wav.shape[-1] / model_sample_rate

        logger.info(f"Audio generated successfully: {request.output_path} (duration: {duration:.2f}s)")
        return {
            "status": "success",
            "output_path": request.output_path,
            "sample_rate": model_sample_rate,
            "duration": duration
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return {
        "status": "healthy",
        "model": "ChatterBoxMultilingualTTS",
        "sample_rate": model_sample_rate,
        "supported_languages": ["en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "ar", "hi", "da", "el", "fi", "he", "ms", "no", "sv", "sw"]
    }

@app.post("/shutdown")
async def shutdown():
    """Shutdown endpoint for graceful termination"""
    logger.info("Shutdown request received")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    logger.info(f"Starting ChatterBox TTS server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 