#!/usr/bin/env python
"""
translator.py

A command-line translator between English and Russian using the Helsinki-NLP
machine translation models.

Usage examples:
  - Command-line arguments:
      python translator.py "Привет, как дела?" ru-en
      python translator.py "Hello, how are you?" en-ru

  - Using pipes:
      echo "Какой сегодня день?" | python translator.py ru-en
      echo "What time is it?" | python translator.py en-ru

  - Interactive mode:
      python translator.py
      (Then follow the prompts; type "exit" to quit.)

This script also uses the following guidelines from Huggingface:
  - https://huggingface.co/Helsinki-NLP/opus-mt-ru-en
  - https://huggingface.co/Helsinki-NLP/opus-mt-en-ru
"""

import os
import sys
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings
warnings.filterwarnings("ignore")

# Maximum chunk size in characters. Adjust this value so that each chunk,
# after tokenization, fits within the model's maximum token limit.
MAX_CHUNK_SIZE = 500

def chunk_text(text, max_length_chars=MAX_CHUNK_SIZE):
    """
    Splits the input text into chunks without breaking sentences.
    
    The function splits the text on sentence-ending punctuation
    (period, exclamation point, question mark) followed by whitespace,
    and then groups sentences into chunks that are at most max_length_chars long.
    
    Args:
      text (str): The input text to be chunked.
      max_length_chars (int): The maximum number of characters allowed per chunk.
      
    Returns:
      list[str]: A list of text chunks.
    """
    # Use a simple regex to split into sentences.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # If adding the sentence would exceed the limit, save the current chunk.
        if current_chunk and (len(current_chunk) + len(sentence) + 1 > max_length_chars):
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = sentence if not current_chunk else current_chunk + " " + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def translate_chunk(text_chunk, model, tokenizer, device):
    """
    Translates a single text chunk using the provided model and tokenizer.
    
    Args:
      text_chunk (str): A chunk of text to translate.
      model: The loaded translation model.
      tokenizer: The corresponding tokenizer.
      device (str): Compute device ("mps", "cuda", or "cpu").
      
    Returns:
      str: The translated text for the chunk.
    """
    # Tokenize the text chunk
    inputs = tokenizer(text_chunk, return_tensors="pt")
    if device in ["cuda", "mps"]:
        inputs = {key: val.to(device) for key, val in inputs.items()}
    # Generate translation for the chunk
    output_tokens = model.generate(**inputs)
    translated_chunk = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_chunk

def translate_long_text(text, model, tokenizer, device, max_length_chars=MAX_CHUNK_SIZE):
    """
    Translates a long text by splitting it into manageable chunks,
    translating each chunk, and then joining the results.
    
    Args:
      text (str): The long input text to translate.
      model: The loaded translation model.
      tokenizer: The corresponding tokenizer.
      device (str): Compute device ("mps", "cuda", or "cpu").
      max_length_chars (int): Maximum number of characters per chunk.
      
    Returns:
      str: The full translated text.
    """
    chunks = chunk_text(text, max_length_chars)
    translations = []
    for chunk in chunks:
        translation = translate_chunk(chunk, model, tokenizer, device)
        translations.append(translation)
    # Join the translated chunks with a space.
    return " ".join(translations)

def get_device():
    """
    Determines which device to use:
      - If mps is available, returns "mps" (for Apple Silicon/MacBook MPS).
      - Else if CUDA is available, returns "cuda".
      - Otherwise, returns "cpu".
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def translate(text, direction, device):
    """
    Translates the input text in the specified direction using the appropriate model.
    
    Args:
      text (str): The text to translate.
      direction (str): Translation direction: "ru-en" for Russian→English,
                       "en-ru" for English→Russian.
      device (str): Compute device ("mps", "cuda", or "cpu").
                       
    Returns:
      str: The translated text.
    """

    start = time.time()

    if direction == "ru-en":
        model_name = "Helsinki-NLP/opus-mt-ru-en"
    elif direction == "en-ru":
        model_name = "Helsinki-NLP/opus-mt-en-ru"
    else:
        sys.exit("Error: Unsupported translation direction. Use 'ru-en' or 'en-ru'.")

    # Force use of the fast tokenizer (which typically avoids a SentencePiece dependency)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move model to the desired device if not CPU
    if device in ["cuda", "mps"]:
        model = model.to(device)

    # If the text is long, translate it in chunks
    if len(text) > MAX_CHUNK_SIZE:
        translated_text = translate_long_text(text, model, tokenizer, device, MAX_CHUNK_SIZE)
    else:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")
        if device in ["cuda", "mps"]:
            inputs = {key: val.to(device) for key, val in inputs.items()}
    
        # Generate translation
        output_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    end = time.time()
    print(f"Translation took {end - start:.2f} seconds.")
    return translated_text

def main():
    device = get_device()

    print("Using device: ", device)
    
    # Case 1: Command-line arguments (e.g. python translator.py "text" direction)
    if len(sys.argv) >= 3:
        text = sys.argv[1]
        direction = sys.argv[2]
        result = translate(text, direction, device)
        print(result)
    # Case 2: Piped input (e.g. echo "text" | python translator.py direction)
    elif not sys.stdin.isatty():
        # When input is piped, use the first command-line argument as the translation direction.
        if len(sys.argv) < 2:
            sys.exit("Error: Please provide the translation direction ('ru-en' or 'en-ru') as an argument.")
        direction = sys.argv[1]
        input_text = sys.stdin.read().strip()
        if not input_text:
            sys.exit("Error: No input text detected from stdin.")
        result = translate(input_text, direction, device)
        print(result)
    # Case 3: Interactive mode (no command-line arguments)
    else:
        print("Interactive mode. Type 'exit' to quit.")
        direction = input("Enter translation direction ('ru-en' or 'en-ru'): ").strip()
        while True:
            text = input("Enter text: ").strip()
            if text.lower() == "exit":
                break
            result = translate(text, direction, device)
            print("Translation:", result)
    

if __name__ == "__main__":
    main()
