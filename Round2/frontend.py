import chainlit as cl
import requests
import json
import os

# Store the last uploaded document ID
current_document_id = None
# Maintain a list of uploaded documents with their file names and IDs
uploaded_documents = {}

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
    global current_document_id, uploaded_documents

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

    # Check if the user wants to list all uploaded documents
    elif message.content == "list":
        if uploaded_documents:
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
        file_name = os.path.basename(file_path)
        query = message.content  # Get user message along with the file

        try:
            # Upload the document first
            with open(file_path, 'rb') as file:
                response = requests.post(url_upload, files={'document': file}, headers={
                    "x-user-id": user_id,
                    "x-session-id": session_id
                })
            response.raise_for_status()  # Raise an error for bad responses

            # Process the response from the server (assuming it returns a document ID)
            response_data = response.json()  # Parse response as JSON
            document_id = response_data.get("document_id", None)

            if document_id:
                # Store the document ID and update the current document ID
                uploaded_documents[file_name] = document_id
                current_document_id = document_id

            # Send a success message to Chainlit about the document upload
            extracted_info = extract_text_JSON(response_data)
            await cl.Message(content=f"Document {file_name} uploaded successfully! {extracted_info}").send()

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
        
        # Extract text from the JSON response
        extracted_text = extract_text_JSON(response_data)

        # Send the extracted text as a response to Chainlit
        await cl.Message(content=extracted_text).send()

    except requests.exceptions.JSONDecodeError:
        # Handle case where response is not valid JSON
        await cl.Message(content="Error: Received non-JSON response from backend.").send()
    except Exception as e:
        await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
