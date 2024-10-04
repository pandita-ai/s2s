## Speech-to-Speech Machine Translation
Speech-to-speech translation, especially in real time, involves computationally heavy ML models that take up a large amount of resources, and building an efficient, cost-effective pipeline is a challenge. The system must process speech quickly and deliver translations with minimal delay.

We built an efficient and fast Speech-to-speech Machine Translation model that leverages GroqCloud’s Fast Inference API to drastically speed up token generation rates compared to other, higher-latency alternatives. 

### Methodology

STT - Audio Transcription (Speech-to-text)

* We use OpenAI’s WhisperV3 model for multilingual text transcription from a source language to English. Whisper is a state-of-the-art, open-source model by OpenAI that handles a wide variety of languages and accents, and it's robust to noise and difficult audio.
* Audio can be provided either from a file upload, or by recording via device microphone. 

MLT - Text Translation (Text-to-text)

* Utilize Llama 3.1-8B-instant through GroqCloud to create a chat completion that translates the transcribed text from English to the target language. 
* Llama 3.1 excels in its multilingual capabilities and wide support for other languages compared to other Groq-compatible LLMs, which makes it a good choice for the translation task. 

TTS - Speech output (Text-to-speech)

* Use OpenAI’s TTS1 model to play back the translated text as realistic speech output.
* Save the speech audio as a file.

### Usage
1) Run the Gradio app using `$python groq_s2s.py`
2) Record or upload audio file (as .wav)
3) Select a target language (audio transcription language defaults to English)
