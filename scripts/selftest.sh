#!/bin/sh
# THE BROWSER ENGINE'S SELF-TEST: the page linked, served with the model's files from $MODEL_DIR (default: the cut
# Qwen3-0.6B in the HuggingFace cache), run in Chromium by scripts/selftest.scala
set -e
cd "$(dirname "$0")/.."
MODEL_DIR=${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--local--Qwen3-0.6B-4bit-cut19/snapshots/main}
sbt -batch browser/fullLinkJS > /dev/null
web=browser/target/web; mkdir -p $web/model
cp browser/index.html $web/ && cp browser/target/scala-3.3.8/okay-llm-qwen-browser-opt/main.js $web/
for f in config.json tokenizer.json model.safetensors; do cp "$MODEL_DIR/$f" $web/model/; done
jwebserver -d "$PWD/$web" -p 8733 -o none > /dev/null 2>&1 &
server=$!
trap 'kill $server' EXIT
sleep 1
scala-cli run scripts/selftest.scala -- "http://localhost:8733/?model=model"
