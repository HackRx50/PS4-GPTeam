import chainlit as cl
import requests
import json
import os
import tempfile  # Import tempfile
from deep_translator import GoogleTranslator  # Use deep-translator
from io import BytesIO
import speech_recognition as sr  # Import SpeechRecognition
from pydub import AudioSegment

# Store the last uploaded document ID
current_document_id = None
# Maintain a list of uploaded documents with their original file names and IDs
uploaded_documents = {}
# Default language is English
current_language = 'en'
# Initialize the deep-translator object (English by default, will change as needed)
translator = GoogleTranslator(source='en', target='en')

def extract_text_JSON(json_object, indent_level=0):
    """Extract text from a JSON object with indentation."""
    lines = []
    indent = '    ' * indent_level  # Create indentation for better readability

    if isinstance(json_object, dict):
        for key, value in json_object.items():
            if isinstance(value, (dict, list)):
                # For nested dictionaries or lists, call the function recursively
                nested_value = extract_text_JSON(value, indent_level + 1)
                lines.append(f'{indent}{key}: {nested_value.strip()}')  # Append key with nested values
            else:
                # Directly append values, with indentation
                lines.append(f'{indent}{key}: {value}')  # Append the key-value pair directly
    elif isinstance(json_object, list):
        for index, item in enumerate(json_object):
            if isinstance(item, (dict, list)):
                # For nested dictionaries or lists, call the function recursively
                nested_value = extract_text_JSON(item, indent_level + 1)
                lines.append(f'{indent}Item {index + 1}: {nested_value.strip()}')  # Append with item index
            else:
                # Directly append values
                lines.append(f'{indent}Item {index + 1}: {item}')  # Append the item directly

    # Join lines into a single string with a period after each line
    return '\n'.join(lines) + '.'  # Ensure there's a period at the end of the string

@cl.on_message
async def on_message(message: cl.Message):
    global current_document_id, uploaded_documents, current_language, translator

    # Check if the message is to set the language
    if message.content.startswith("set language "):
        lang = message.content[len("set language "):].strip().lower()
        # Update the current language based on user input
        current_language = lang
        translator = GoogleTranslator(source='en', target=current_language)  # Update the translator's target language
        await cl.Message(content=f"Language set to {lang}").send()
        return  # Exit the function after setting the language

    # URLs for your Flask server
    url_chat = "http://localhost:3000/chat"
    url_upload = "http://localhost:3000/upload_document"

    # Prepare headers with user and session IDs
    user_id = "123"  # Replace with the actual user ID
    session_id = "J5K7P1ZQ"  # Replace with the actual session ID

    # Check if the message is to switch documents
    if message.content.startswith("switch to "):
        file_name = message.content[len("switch to "):].strip()
        if file_name in uploaded_documents:
            current_document_id = uploaded_documents[file_name]
            await cl.Message(content=f"Switched to document {file_name} with ID {current_document_id}").send()
        else:
            await cl.Message(content=f"Document {file_name} not found.").send()
        return  # Exit the function after switching
    elif message.content == "list":
        if uploaded_documents:
            await cl.Message(content=f"Uploaded documents: {', '.join(uploaded_documents.keys())}").send()
            document_list = "\n".join([f"{name}: {doc_id}" for name, doc_id in uploaded_documents.items()])
            await cl.Message(content=f"Uploaded documents:\n{document_list}").send()
        else:
            await cl.Message(content="No documents uploaded.").send()
        return  # Exit the function after listing

    # Check if there are any files attached
    if message.elements:
        # Processing the first attached document (if any)
        document = message.elements[0]  # Assuming only one file
        file_path = document.path
        original_file_name = os.path.basename(file_path)  # Get the original file name
        print(original_file_name)
        query = message.content  # Get user message along with the file

        try:
            # Upload the document first
            with open(file_path, 'rb') as file:
                response = requests.post(url_upload, files={'document': file}, headers={
                    "x-user-id": user_id,
                    "x-session-id": session_id
                })
            response.raise_for_status()  # Raise an error for bad responses

            # Assuming the server will return the document ID in the response
            response_data = response.json()  # Parse response as JSON
            document_id = response_data.get("document_id", None)

            if document_id:
                # Store the original filename and its document ID
                uploaded_documents[original_file_name] = document_id
                current_document_id = document_id  # Update current document ID

            # Send a success message to Chainlit about the document upload
            extracted_info = extract_text_JSON(response_data)
            await cl.Message(content=f"Document {original_file_name} uploaded successfully! {extracted_info}").send()

            # Send the chat message with the document ID
            await send_chat_message(url_chat, query, document_id, user_id, session_id)

        except requests.exceptions.RequestException as e:
            await cl.Message(content=f"An error occurred while uploading the document: {str(e)}").send()
            return  # Stop further processing if upload fails
        except Exception as e:
            await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()
            return  # Stop further processing if there's any other error

    else:
        # If no files are attached, proceed with normal chat processing
        query = message.content  # Get user message
        if current_document_id:
            query += f"\n(Document ID: {current_document_id})"
        
        await send_chat_message(url_chat, query, current_document_id, user_id, session_id)

async def send_chat_message(url_chat, query, document_id, user_id, session_id):
    """Send the chat message to the server."""
    try:
        # Send a POST request to the Flask server for chat using requests
        response = requests.post(url_chat, json={
            "query": query,
            "document_id": document_id  # Include the document ID (if any)
        }, headers={
            "Content-Type": "application/json",
            "x-user-id": user_id,
            "x-session-id": session_id
        })
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response from the Flask server
        response_data = response.json()  # Ensure the response is parsed as JSON

        # Extract the bot message and translate it
        bot_message = response_data.get("bot_message", "")
        if current_language != 'en':  # If the language is not English, translate the message
            translated_bot_message = translator.translate(bot_message)
        else:
            translated_bot_message = bot_message

        # Extract text from the JSON response
        extracted_text = extract_text_JSON(response_data)

        # Send the translated bot message as a response to Chainlit
        await cl.Message(content=translated_bot_message).send()

    except requests.exceptions.JSONDecodeError:
        # Handle case where response is not valid JSON
        await cl.Message(content="Error: Received non-JSON response from backend.").send()
    except Exception as e:
        await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handle incoming audio chunk from the user's microphone."""
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for speech_recognition to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements):
    """Handle the end of audio recording and transcribe the audio."""
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    # Transcribe the audio
    transcribed_text = await transcribe_audio(audio_file, audio_mime_type)

    # Send the transcribed text back to the user
    await cl.Message(content=f"Transcribed text: {transcribed_text}").send()

def convert_audio_format(input_path, output_path, target_format="wav"):
    """Convert audio to the target format using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=target_format)

async def transcribe_audio(audio_file, audio_mime_type):
    """Transcribe audio to text using the SpeechRecognition library."""
    recognizer = sr.Recognizer()

    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        temp_audio_file.write(audio_file)

    # Convert the audio to WAV format
    wav_audio_path = temp_audio_path.replace(".mp4", ".wav")
    convert_audio_format(temp_audio_path, wav_audio_path)

    try:
        with sr.AudioFile(wav_audio_path) as source:
            audio_data = recognizer.record(source)
            result = recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Transcription failed."

    # Clean up the temporary audio files
    os.remove(temp_audio_path)
    os.remove(wav_audio_path)

    return result
