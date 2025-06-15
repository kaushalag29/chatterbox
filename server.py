import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
import os

app = FastAPI(title="Chatterbox TTS API")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

# Patch torch.load to use the correct device
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

# Load model at startup
model = ChatterboxTTS.from_pretrained(device=device)

# Define request model
class TTSRequest(BaseModel):
    subtitle_text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5

@app.post("/generate-audio")
async def generate_audio(request: TTSRequest):
    try:
        print(f"Generating audio for subtitle - {request.subtitle_text}, exaggeration - {request.exaggeration}, cfg_weight - {request.cfg_weight}")
        # Fixed audio prompt path
        audio_prompt_path = "./../OpenVoice/target000.wav"
        
        # Check if the audio prompt file exists
        if not os.path.exists(audio_prompt_path):
            raise HTTPException(status_code=400, detail=f"Audio prompt file not found at {audio_prompt_path}")
        print(f"Audio prompt file found at {audio_prompt_path}")

        # Generate audio
        wav = model.generate(
            request.subtitle_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight
        )
        
        # Save the generated audio
        output_path = "./../OpenVoice/cloned_audio.wav"
        print(f"Saving audio to {output_path}")
        ta.save(output_path, wav, model.sr)
        
        return {"status": "success", "message": f"Audio saved to {output_path}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 