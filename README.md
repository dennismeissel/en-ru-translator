# English-Russian Translator

A command-line translator between **English and Russian**, using **Helsinki-NLP** models.

- **ru-en**: Russian → English (`Helsinki-NLP/opus-mt-ru-en`)
- **en-ru**: English → Russian (`Helsinki-NLP/opus-mt-en-ru`)

## Installation


### **1. Clone the Repository**
```sh
git clone https://github.com/YOUR_USERNAME/english-russian-translator.git
cd english-russian-translator
```

### **2. Install Dependencies**

On MacOS:
```sh
brew install cmake pkg-config
```

```sh
pip install -r requirements.txt
```


## Usage

### **1. Command-line Arguments**
```sh
python translator.py "Привет, как дела?" ru-en
python translator.py "Hello, how are you?" en-ru
```

### **2. Using Pipes**
```sh
echo "Какой сегодня день?" | python translator.py ru-en
echo "What time is it?" | python translator.py en-ru
```

### **3. Interactive Mode**
```sh
python translator.py
```
```
Type text and press Enter. Type `exit` to quit.
```

## Credits

This project uses **Helsinki-NLP** machine translation models:

- **[Helsinki-NLP/opus-mt-ru-en](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en)**
- **[Helsinki-NLP/opus-mt-en-ru](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru)**

Developed by the **Language Technology Research Group at the University of Helsinki**,  
licensed under **CC BY 4.0**.

## License

Licensed under **CC BY 4.0**. See [`LICENSE`](LICENSE).
