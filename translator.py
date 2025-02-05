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
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_device():
    """
    Determines which device to use:
      - If mps is available,
        returns "mps" (for Apple Silicon/MacBook MPS).
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
    if direction == "ru-en":
        model_name = "Helsinki-NLP/opus-mt-ru-en"
    elif direction == "en-ru":
        model_name = "Helsinki-NLP/opus-mt-en-ru"
    else:
        sys.exit("Error: Unsupported translation direction. Use 'ru-en' or 'en-ru'.")

    # Load tokenizer and model from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move model to the desired device if not CPU
    if device in ["cuda", "mps"]:
        model = model.to(device)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    if device in ["cuda", "mps"]:
        inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate translation
    output_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

def main():
    device = get_device()
    
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
