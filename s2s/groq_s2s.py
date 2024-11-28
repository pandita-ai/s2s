import os
import gradio as gr
from groq import Groq
from openai import OpenAI

api_key_g = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key_g)

api_key_a = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key_a)

# Arbitrary list of languages to translate to

languages = {
   "Spanish": "es",
   "German": "de",
   "Portuguese": "pt",
   "French": "fr",
   "Italian": "it",
   "Russian": "ru",
   "Japanese": "ja",
   "Chinese": "zh",
   "Korean": "ko",
   "Hindi": "hi"
}

def transcribe_and_translate(audio_filepath, target_language):
    filename = audio_filepath
    
    # Transcribe audio using Groq API and WhisperV3
    with open(filename, "rb") as file:
        transcription = client.audio.translations.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            response_format="json",
            temperature=0.2
        )

    transcribed_text = transcription.text
    
    # Translate text using groq_translate function (source language is hardcoded to English for now)
    translated_text = groq_translate(transcribed_text, "English", target_language)

    return transcribed_text, translated_text

# Function to handle the translation step
def groq_translate(query, from_language, to_language):

    # Creates a Llama 3.1 chat completion to output translated text

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a knowledgeable translation assistant that translates text from {from_language} to {to_language}. "
                           f"You will only reply with the translated text in {to_language}, and nothing else."
            },
            {
                "role": "user",
                "content": f"Translate the following from {from_language} to {to_language}: '{query}'"
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        stream=False,
    )

    return chat_completion.choices[0].message.content

# Function to generate speech from the translated text using OpenAI TTS1
def generate_speech(text):
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    
    # Save the TTS output to an audio file
    output_filename = "output.mp3"
    response.stream_to_file(output_filename)

    return output_filename

# Interface using Gradio
def interface():
    language_dropdown = gr.Dropdown(choices=list(languages.keys()), label="Select language to translate to:")
    audio_input = gr.Audio(sources=['upload', 'microphone'], type="filepath", label="Record your audio")

    output_transcription = gr.Textbox(label="Transcribed Text: English")
    output_translation = gr.Textbox(label=f"Translated Text:")

    tts_output = gr.Audio(label="Text-to-Speech Output")

    def process_audio(audio_filepath, target_language):
        transcribed_text, translated_text = transcribe_and_translate(audio_filepath, target_language)
        tts_audio_file = generate_speech(translated_text)
        
        return transcribed_text, translated_text, tts_audio_file

    gr.Interface(
        fn=process_audio,
        inputs=[audio_input, language_dropdown],
        outputs=[output_transcription, output_translation, tts_output],
        title="Groq Speech Translator with TTS",
        description="Record audio, translate it into a selected language, and convert the translation to speech using OpenAI's TTS."
    ).launch(share=True)

if __name__ == "__main__":
    interface()
