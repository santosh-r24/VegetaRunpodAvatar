# Vegeta Avatar Demo - RunPod Worker
Worker template for RunPod serverless that generates AI avatar videos from audio input using MuseTalk and VITS TTS models.

## 🔧 Technical Stack

- **Avatar Generation**: MuseTalk (facial animation) for lip synch
- **TTS Model**: Custom VITS trained model on Vegeta's voice 
- **Infrastructure**: RunPod Serverless GPU + HF spaces

## Docker Image
```
docker image: santoshr24/musetalk_vegeta_endpoint:latest
```

## Payload
```
{"input": {
                "audio_base64": base_64_encoded_audio,
                "audio_name": filename
            }}
```
## Response
```
{"video_base64": video_b64, "video_name": filename}
```