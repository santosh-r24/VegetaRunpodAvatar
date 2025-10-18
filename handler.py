import runpod 
import torch
import base64
import subprocess
import os
import pickle
import glob
import copy
import cv2
import uuid
import numpy as np
import subprocess
import logging
import time
from pathlib import Path
from transformers import WhisperModel
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, '/workspace/MuseTalk')
from MuseTalk.musetalk.utils.utils import load_all_model
from MuseTalk.musetalk.utils.preprocessing import read_imgs
from MuseTalk.musetalk.utils.audio_processor import AudioProcessor
from MuseTalk.musetalk.utils.face_parsing import FaceParsing
from MuseTalk.musetalk.utils.utils import datagen
from MuseTalk.musetalk.utils.blending import get_image_blending

def setup_logger(log_level=logging.DEBUG):
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - [Request: %(request_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("runpod_worker")
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger

logger = setup_logger(log_level=logging.DEBUG)
logger = logging.LoggerAdapter(logger, {"request_id": "N/A"})

#loading Models
print("\n[INFO] Loading models.")
ROOT_DIR = "/workspace/MuseTalk"
FPS = 25
paths = {
        "musetalk_json": os.path.join(ROOT_DIR, "models/musetalk_v15/musetalk.json"),
        "musetalk_unet": os.path.join(ROOT_DIR, "models/musetalk_v15/unet.pth"),
        "vae_config": os.path.join(ROOT_DIR, "models/sd-vae-ft-mse/config.json"),
        "vae_weights": os.path.join(ROOT_DIR, "models/sd-vae-ft-mse/diffusion_pytorch_model.bin"),
        "whisper": os.path.join(ROOT_DIR, "models/whisper/pytorch_model.bin"),
        "whisper_config":os.path.join(ROOT_DIR, "models/whisper/config.json"),
        "whisper_preprocessor_config":os.path.join(ROOT_DIR, "models/whisper/preprocessor_config.json"),
        "whisper_tf_model":os.path.join(ROOT_DIR, "models/whisper/tf_model.h5"),
        "whisper_pt":os.path.join(ROOT_DIR, "models/whisper/tiny.pt"),
        "dwpose": os.path.join(ROOT_DIR, "models/dwpose/dw-ll_ucoco_384.pth"),
        "bisenet": os.path.join(ROOT_DIR, "models/face-parse-bisent/79999_iter.pth"),
        "bisenet_backbone": os.path.join(ROOT_DIR, "models/face-parse-bisent/resnet18-5c106cde.pth"),
        } #Change line to network volume
AVATAR_ID = "vegeta"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
        unet_model_path=paths["musetalk_unet"],
        vae_type="sd-vae-ft-mse",
        unet_config=paths["musetalk_json"],
    )
pe = pe.half().to(device)
unet.model = unet.model.half().to(device)
vae.vae = vae.vae.half().to(device)
weight_dtype = unet.model.dtype
whisper = WhisperModel.from_pretrained("/workspace/MuseTalk/models/whisper") #Change line to network volume
whisper = whisper.to(device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)
audio_processor = AudioProcessor(feature_extractor_path="/workspace/MuseTalk/models/whisper") #Change line to network volume
fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

#loading data
tmp_avatar_dir = f"/workspace/MuseTalk/{AVATAR_ID}/"
full_imgs_dir = os.path.join(tmp_avatar_dir, "full_imgs") #Change line to network volume
mask_dir = os.path.join(tmp_avatar_dir, "mask") #Change line to network volume
tmp_frames_dir = os.path.join("/workspace/", "tmp")
os.makedirs(tmp_frames_dir, exist_ok=True)

coords_path = os.path.join(tmp_avatar_dir, "coords.pkl") #Change line to network volume
latents_path = os.path.join(tmp_avatar_dir, "latents.pt") #Change line to network volume
mask_coords_path = os.path.join(tmp_avatar_dir, "mask_coords.pkl") #Change line to network volume

print("\n[INFO] Using cached avatar latents and coordinates.")
with open(coords_path, "rb") as f:
    coords = pickle.load(f)
with open(mask_coords_path, "rb") as f:
    mask_coords = pickle.load(f)
frames = read_imgs(sorted(glob.glob(f"{full_imgs_dir}/*.png")))
masks = read_imgs(sorted(glob.glob(f"{mask_dir}/*.png")))
latents = torch.load(latents_path)

def save_blended_frame(i, res_frame, coords, frames, masks, mask_coords, output_dir):
    bbox = coords[i % len(coords)]
    x1, y1, x2, y2 = bbox
    ori_frame = copy.deepcopy(frames[i % len(frames)])
    try:
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
    except Exception as e:
        return
    mask = masks[i % len(masks)]
    mask_box = mask_coords[i % len(mask_coords)]
    blended = get_image_blending(ori_frame, res_frame, bbox, mask, mask_box)
    cv2.imwrite(os.path.join(output_dir, f"{i:08d}.png"), blended)

def generate_video(audio, frames_dir):
    video_id = uuid.uuid4().hex
    video_path = f"/workspace/vegeta_video_encoded_{video_id}.ts"
    subprocess.run([
    "ffmpeg", "-y", "-r", str(FPS),
    "-i", f"{frames_dir}/%08d.png",
    "-i", audio,
    "-c:v", "libx264",  # encode video
    "-pix_fmt", "yuv420p",
    "-shortest",
    "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
    "-f", "mpegts",     # MPEG-TS container
    video_path
], check=True)
    return video_path

def handler(job):
    """Handler function that will be used to process jobs."""
    request_id = job.get('id', 'unknown')
    job_logger = logging.LoggerAdapter(logging.getLogger("runpod_worker"), {"request_id": request_id})

    start_time = time.time()
    job_logger.info("Handler started.")

    audio_b64 = job["input"]["audio_base64"]
    audio_bytes = base64.b64decode(audio_b64)
    audio_name = job["input"]["audio_name"]
    audio_path = f"/workspace/{audio_name}"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    print(f"[INFO] Audio byte size: {len(audio_bytes)} bytes")
    output_dir = os.path.join("/workspace/", "tmp", Path(os.path.basename(audio_name)).stem)
    os.makedirs(output_dir, exist_ok=True)
    
    # Example: timing a function
    func_start = time.time()
    print(f"\n [INFO] Processing audio path {audio_path} and generating video...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=FPS, audio_padding_length_left=2, audio_padding_length_right=2
    )
    func_end = time.time()
    job_logger.info(f"Function 'get_whisper_chunks' took {func_end - func_start:.3f} seconds.")
    
    print("\n[INFO] Starting generation of images...")
    func_start = time.time()
    gen = datagen(whisper_chunks, latents, batch_size=15)
    idx = 0
    for whisper_batch, latent_batch in gen:
        audio_features = pe(whisper_batch.to(device))
        latents_tensor = latent_batch.to(device=device, dtype=weight_dtype)
        preds = unet.model(latents_tensor, torch.tensor([0], device=device), encoder_hidden_states=audio_features).sample
        preds = preds.to(device=device, dtype=weight_dtype)
        decoded_frames = vae.decode_latents(preds)
        for res_frame in decoded_frames:
            save_blended_frame(idx, res_frame, coords, frames, masks, mask_coords, output_dir)
            idx += 1
        
    func_end = time.time()
    job_logger.info(f"Function 'generating and saving images' took {func_end - func_start:.3f} seconds.")
    
    print("\n[INFO] Starting generation of video...")
    func_start = time.time()
    video_path = generate_video(audio_path, output_dir)
    with open(video_path, "rb") as vf:
        video_bytes = vf.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    func_end = time.time()
    job_logger.info(f"Function 'generate and encoding video' took {func_end - func_start:.3f} seconds.")
    
    print(f"\n[INFO] Video byte size: {len(video_bytes)} bytes | Video base64 size: {len(video_b64)} characters")
    print(f"\n[INFO] Sending video {os.path.basename(video_path)}")  
    end_time = time.time()
    job_logger.info(f"\nTotal handler time: {end_time - start_time:.3f} seconds.")  
    return {"video_base64": video_b64, "video_name": os.path.basename(video_path)}

runpod.serverless.start({"handler": handler})
