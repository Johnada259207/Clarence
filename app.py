import os
import sqlite3
import json
from openai import OpenAI
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# We still have two models, but we now unify them in one approach
RESOURCE_MODEL = "deepseek-reasoner"
CONVERSATION_MODEL = "deepseek-chat"

client = OpenAI(
    api_key="sk-3709591f17a94be08dde9fe50e28fc8a", 
    base_url="https://api.deepseek.com"
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

def fetch_all_resources():
    """Retrieve all database resources."""
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
    user_input = request.form.get("query", "").strip()
    if not user_input:
        return jsonify({"response": "I didn't catch that. Please try again."})

    # 1. Add user message to conversation history
    session["conversation"].append({"role": "user", "content": user_input})

    # 2. Resource detection
    resource_triggers = [
        "resource", "resources", "bus pass", "transport", "shelter", 
        "emergency", "ride", "transportation", "assistance", "food bank", "food banks"
    ]
    is_resource_query = any(trigger in user_input.lower() for trigger in resource_triggers)
    resource_text = ""
    if is_resource_query:
        # If resource query, fetch from DB
        resources = fetch_all_resources()
        if resources:
            # If you have many resources, the AI might produce very long answers.
            # Let's limit to top 5 for brevity:
            resources = resources[:5]
            resource_text = "\n".join([f"{org} - {desc}" for org, desc in resources])

    # 3. Build system + conversation messages
    # We ask the AI to produce JSON with "topic" and "answer".
    # We instruct it NOT to use triple backticks or code fences.
    system_instructions = (
        "You are a helpful AI assistant. Maintain conversation context using the messages below.\n"
        "If the user is asking about resources, only use the provided resource data. If not, be conversational.\n\n"
        "IMPORTANT: Return your response as valid JSON with two keys: \"topic\" and \"answer\".\n"
        "Example:\n"
        "{\"topic\": \"Transportation Assistance\", \"answer\": \"Here is the info...\"}\n\n"
        "Do NOT include triple backticks or code fences. Only return valid JSON.\n"
        "Keep your answers concise. If the user wants more details, ask them.\n"
        "If the answer is near the token limit, summarize and mention you can provide more if requested.\n"
        "Do not include any source attributions beyond the resource text provided.\n"
    )

    # Add resource data to the user content if relevant
    user_context = user_input
    if is_resource_query and resource_text:
        user_context += (
            f"\n\nRelevant resources (limit to top 5 if many):\n{resource_text}"
        )

    # Combine the system message + conversation history
    messages = [{"role": "system", "content": system_instructions}]
    for msg in get_trimmed_history():
        messages.append(msg)
    messages.append({"role": "user", "content": user_context})

    # 4. Attempt call with resource or conversation model
    model_to_use = RESOURCE_MODEL if is_resource_query else CONVERSATION_MODEL

    def call_deepseek(model_name, msg_list):
        return client.chat.completions.create(
            model=model_name,
            messages=msg_list,
            stream=False,
            max_tokens=600  # Increase if you want more content
        )

    try:
        response = call_deepseek(model_to_use, messages)
        assistant_raw = response.choices[0].message.content
    except Exception as e:
        print(f"Error with DeepSeek API (primary call): {e}")
        # Attempt fallback with conversation model
        try:
            fallback_response = call_deepseek(CONVERSATION_MODEL, messages)
            assistant_raw = fallback_response.choices[0].message.content
        except Exception as e2:
            print(f"Error with fallback call: {e2}")
            assistant_raw = '{"topic": "Error", "answer": "Sorry, I am having trouble responding right now."}'

    # 5. Parse the JSON the AI returned
    topic, answer = parse_json_assistant_response(assistant_raw)

    # 6. Update the topic in session
    session["current_topic"] = topic

    # 7. Add the AI response to conversation history
    session["conversation"].append({"role": "assistant", "content": answer})

    return jsonify({"response": answer, "topic": topic})

def parse_json_assistant_response(raw_text):
    """
    Attempt to parse the AI's response as JSON with 'topic' and 'answer'.
    If it fails, fallback to a generic result.
    """
    topic = session.get("current_topic", "General Conversation")
    answer = "Sorry, I couldn't parse the response."

    # Remove any stray backticks if the model included them
    cleaned_text = raw_text.replace("```", "").strip()

    try:
        # Attempt to locate the JSON braces
        start_idx = cleaned_text.find("{")
        end_idx = cleaned_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_text[start_idx:end_idx+1]
            data = json.loads(json_str)
            topic = data.get("topic", topic)
            answer = data.get("answer", answer)
        else:
            # If we can't find braces, treat the entire text as 'answer'
            answer = cleaned_text
    except Exception as e:
        print(f"JSON parse error: {e}")
        # We'll assume the entire raw_text is just the "answer"
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
