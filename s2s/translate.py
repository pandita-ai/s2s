from openai import OpenAI
from groq import Groq
import gradio as gr
Aimport os

# Initialize the API keys for Groq and OpenAI services
api_key_g = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key_g)

api_key_a = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key_a)

# List of languages for translation with language codes
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

# Function to transcribe audio and translate the transcribed text to a
# target language


def transcribe_and_translate(
        audio_filepath: str,
        target_language: str) -> tuple:
    """
    Transcribes the provided audio file using Groq's WhisperV3 model and translates the transcribed text
    to the specified target language.

    :param audio_filepath: Path to the audio file to transcribe.
    :param target_language: The target language for the translation (in string format, e.g., "Spanish").
    :return: A tuple containing the transcribed text and the translated text.
    """
    filename = audio_filepath

    # Open and transcribe the audio file using Groq Whisper API
    with open(filename, "rb") as file:
        transcription = client.audio.translations.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            response_format="json",
            temperature=0.2
        )

    transcribed_text = transcription.text

    # Translate the transcribed text to the target language
    translated_text = groq_translate(
        transcribed_text, "English", target_language)

    return transcribed_text, translated_text

# Function to translate text from one language to another using Llama 3.1 model


def groq_translate(query: str, from_language: str, to_language: str) -> str:
    """
    Translates a given query from one language to another using the Groq client and Llama 3.1 model.

    :param query: The text that needs to be translated.
    :param from_language: The source language of the query text.
    :param to_language: The target language to translate the query into.
    :return: Translated text in the target language.
    """

    # Create a chat completion with the Llama model to generate the translation
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a knowledgeable translation assistant that translates text from {from_language} to {to_language}. "
                f"You will only reply with the translated text in {to_language}, and nothing else."},
            {
                "role": "user",
                "content": f"Translate the following from {from_language} to {to_language}: '{query}'"}],
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        stream=False,
    )

    # Return the translated text
    return chat_completion.choices[0].message.content

# Function to convert the translated text to speech using OpenAI TTS1


def generate_speech(text: str) -> str:
    """
    Generates speech from the provided text using OpenAI TTS1 model and saves it to an audio file.

    :param text: The text to convert into speech.
    :return: The file path of the generated speech audio.
    """

    # Create a speech synthesis request
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    # Save the synthesized speech to an output audio file
    output_filename = "output.mp3"
    response.stream_to_file(output_filename)

    return output_filename

# Function to define the user interface and interaction flow for the Gradio app


def interface():
    """
    Defines the Gradio interface for the speech-to-speech translator. It includes audio input, language selection,
    and buttons for transcription, translation, and text-to-speech functionality.
    """
    with gr.Blocks() as demo:
        gr.Markdown("""
        <div style="position: absolute; top: 10px; right: 20px; z-index: 10;">
            <img src="/file=static/logo_white.png" alt="Logo" style="width: 150px;"/>
        </div>
        """)

        # Define input and output components for the interface
        language_dropdown = gr.Dropdown(
            choices=list(
                languages.keys()),
            label="Select language to translate to:")
        audio_input = gr.Audio(
            sources=[
                'upload',
                'microphone'],
            type="filepath",
            label="Record your audio")

        # Textboxes for displaying transcribed and translated text
        output_transcription = gr.Textbox(label="Transcribed Text: English")
        output_translation = gr.Textbox(label="Translated Text:")

        # Output for synthesized speech
        tts_output = gr.Audio(label="Text-to-Speech Output")

        # Function to process audio input, perform transcription, translation,
        # and TTS
        def process_audio(audio_filepath: str, target_language: str) -> tuple:
            """
            Handles the processing of the audio input, including transcription, translation, and
            generating speech from the translated text.

            :param audio_filepath: The file path of the uploaded/recorded audio.
            :param target_language: The target language for translation.
            :return: A tuple containing the transcribed text, translated text, and TTS audio file path.
            """
            transcribed_text, translated_text = transcribe_and_translate(
                audio_filepath, target_language)
            tts_audio_file = generate_speech(translated_text)
            return transcribed_text, translated_text, tts_audio_file

        # Interface layout with input/output interaction
        gr.Interface(
            fn=process_audio,
            title="Speech2Speech Machine Translator (GroqCloud)",
            inputs=[audio_input, language_dropdown],
            outputs=[output_transcription, output_translation, tts_output],
            live=False  # Disable live mode, the process starts only when button is clicked
        )

    # Launch the Gradio demo with specific allowed file paths for static assets
    demo.launch(allowed_paths=["./../static"])


# Run the interface when this script is executed
if __name__ == "__main__":
    interface()
