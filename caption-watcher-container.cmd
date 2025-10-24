echo off
export CUDA_VISIBLE_DEVICES=0
echo [Watcher] Starting caption watcher...
docker run -d  --name caption-watcher --gpus all -v "dataset/captioning/mainchar" -v D:\scripts:/scripts caption-watcher:latest