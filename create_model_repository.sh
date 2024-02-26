#!/bin/bash

# Define the target directory where the files will be created
target_dir="./models_repository/generator"

# Create the directory structure
mkdir -p "$target_dir/tokenizer_encoder/1"
mkdir -p "$target_dir/tokenizer_decoder/1"
mkdir -p "$target_dir/generator_model/1"
mkdir -p "$target_dir/ensemble_model/1"

# Create the files using touch
touch "$target_dir/tokenizer_encoder/1/merges.txt"
touch "$target_dir/tokenizer_encoder/1/config.json"
touch "$target_dir/tokenizer_encoder/1/model.py"
touch "$target_dir/tokenizer_encoder/1/vocab.json"
touch "$target_dir/tokenizer_decoder/1/merges.txt"
touch "$target_dir/tokenizer_decoder/1/config.json"
touch "$target_dir/tokenizer_decoder/1/model.py"
touch "$target_dir/tokenizer_decoder/1/vocab.json"
touch "$target_dir/tokenizer_encoder/config.pbtxt"
touch "$target_dir/tokenizer_decoder/config.pbtxt"
touch "$target_dir/generator_model/config.pbtxt"
touch "$target_dir/ensemble_model/config.pbtxt"
