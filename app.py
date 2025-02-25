import os
import sqlite3
import json
import openai  # Standard openai library
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("Warning: DEEPSEEK_API_KEY not found in environment variables.")

# Set the key/base for openai (DeepSeek if openai-compatible)
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com"  # If your DeepSeek endpoint is truly openai-compatible

# Use only the deepseek-chat model for all queries
CHAT_MODEL = "deepseek-chat"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")


def fetch_all_resources():
    """Retrieve all database resources from SQLite."""
    try:
        conn = sqlite3.connect("resources.db")
        cursor = conn.cursor()
        cursor.execute("SELECT organization_name, short_description FROM resources")
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()


def get_trimmed_history(max_messages=10):
    """Return the most recent messages from conversation history."""
    return session["conversation"][-max_messages:]


@app.before_request
def ensure_context():
    """Ensure session variables exist before each request."""
    if "conversation" not in session:
        session["conversation"] = []
    if "current_topic" not in session:
        session["current_topic"] = "General Conversation"


@app.route("/")
def index():
    return render_template("index.html", current_topic=session["current_topic"])


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint: handles user queries and returns AI responses in JSON."""
    user_input = request.form.get("query", "").strip()
    if not user_input:
        return jsonify({"response": "I didn't catch that. Please try again."})

    # Add the user's message to the conversation history
    session["conversation"].append({"role": "user", "content": user_input})

    # Detect if the query is resource-related
    resource_triggers = [
        "resource", "resources", "bus pass", "transport", "shelter",
        "emergency", "ride", "transportation", "assistance",
        "food bank", "food banks"
    ]
    is_resource_query = any(trigger in user_input.lower() for trigger in resource_triggers)

    # If resource query, gather resource data
    resource_text = ""
    if is_resource_query:
        resources = fetch_all_resources()
        if resources:
            # Limit to top 5 to avoid super-long answers
            resources = resources[:5]
            resource_text = "\n".join([f"{org} - {desc}" for org, desc in resources])

    # Build the system instructions — we call the AI "Clarence"
    system_instructions = (
        "You are Clarence, a helpful clinical AI assistant. Maintain conversation context using the messages below.\n"
        "If the user is asking about resources, only use the provided resource data. If not, be conversational.\n\n"
        "IMPORTANT: Return your response as valid JSON with two keys: \"topic\" and \"answer\".\n"
        "Example:\n"
        "{\"topic\": \"Transportation Assistance\", \"answer\": \"Here is the info...\"}\n\n"
        "Do NOT include triple backticks or code fences. Only return valid JSON.\n"
        "Keep your answers concise. If the user wants more details, ask them.\n"
        "If the answer is near the token limit, summarize and mention you can provide more if requested.\n"
        "Do not include any source attributions beyond the resource text provided.\n"
    )

    # Append resource data if relevant
    user_context = user_input
    if is_resource_query and resource_text:
        user_context += f"\n\nRelevant resources (limit to top 5 if many):\n{resource_text}"

    # Combine system + conversation messages
    messages = [{"role": "system", "content": system_instructions}]
    for msg in get_trimmed_history():
        messages.append(msg)
    messages.append({"role": "user", "content": user_context})

    # We now always use deepseek-chat
    def call_deepseek(msg_list):
        return openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=msg_list,
            max_tokens=600  # Adjust as needed
        )

    # Try the single model call
    try:
        response = call_deepseek(messages)
        assistant_raw = response.choices[0].message.content
    except Exception as e:
        print(f"Error with DeepSeek API: {e}")
        # If it fails, produce an error JSON
        assistant_raw = '{"topic": "Error", "answer": "Sorry, I am having trouble responding right now."}'

    # Parse the JSON
    topic, answer = parse_json_assistant_response(assistant_raw)

    # Update topic and conversation
    session["current_topic"] = topic
    session["conversation"].append({"role": "assistant", "content": answer})

    return jsonify({"response": answer, "topic": topic})


def parse_json_assistant_response(raw_text):
    """
    Attempt to parse the AI's response as JSON with 'topic' and 'answer'.
    If it fails, fallback to a generic result.
    """
    topic = session.get("current_topic", "General Conversation")
    answer = "Sorry, I couldn't parse the response."

    cleaned_text = raw_text.replace("```", "").strip()
    try:
        start_idx = cleaned_text.find("{")
        end_idx = cleaned_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_text[start_idx:end_idx+1]
            data = json.loads(json_str)
            topic = data.get("topic", topic)
            answer = data.get("answer", answer)
        else:
            answer = cleaned_text
    except Exception as e:
        print(f"JSON parse error: {e}")
        answer = cleaned_text

    return topic, answer


@app.route("/reset_topic", methods=["POST"])
def reset_topic():
    """Clears the conversation history and resets the current topic."""
    session["conversation"] = []
    session["current_topic"] = "General Conversation"
    return jsonify({"current_topic": session["current_topic"]})


if __name__ == "__main__":
    app.run(debug=True)
