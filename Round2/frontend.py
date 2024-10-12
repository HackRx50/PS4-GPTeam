import chainlit as cl
import requests
import json

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

def extract_text_JSON_from_file(file_path):
    """Extract text from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as json_file:
        # Load the entire JSON file
        json_object = json.load(json_file)  # This will load the whole JSON object into memory
        return extract_text_JSON(json_object)

@cl.on_message
async def on_message(message: cl.Message):
    # URL for your Flask server
    url_chat = "http://localhost:3000/chat"

    # Prepare headers with user and session IDs
    user_id = "123"  # Replace with the actual user ID
    session_id = "J5K7P1ZQ"  # Replace with the actual session ID

    # Prepare the query and document ID
    query = message.content  # Get user message from the Message object
    document_id = "doc_fa2b54f9a1e8"  # Replace with the actual document ID if necessary

    # Construct the curl command string using f-string formatting
    curl_command = (
        f"curl --location '{url_chat}' "
        f"--header 'Content-Type: application/json' "
        f"--header 'x-user-id: {user_id}' "
        f"--header 'x-session-id: {session_id}' "
        f"--data '{{\"query\": \"{query}\", \"document_id\": \"{document_id}\"}}'"
    )

    # Print the constructed curl command
    print("CURL Command:", curl_command)

    try:
        # Send a POST request to the Flask server for chat using requests
        response = requests.post(url_chat, json={
            "query": query,
            "document_id": document_id
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
        print("Extracted text:", extracted_text)
        # Send the extracted text as a response to Chainlit
        await cl.Message(content=extracted_text).send()

    except requests.exceptions.JSONDecodeError:
        # Handle case where response is not valid JSON
        await cl.Message(content="Error: Received non-JSON response from backend.").send()
    except Exception as e:
        await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
