import os
from dotenv import load_dotenv
from google import genai
import json
import random

load_dotenv()

# 1. Initialize client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# 2. Define available tools
TOOLS = {
    "get_weather": {
        "description": "Get current weather for a city",
        "parameters": {"city": "string"}
    },
    "get_joke": {
        "description": "Tell a random programming joke",
        "parameters": {}
    }
}

# 3. Implement tools
def get_weather(city):
    data = {
        "Tel Aviv": "27°C and sunny",
        "London": "15°C and rainy",
        "New York": "22°C and cloudy"
    }
    return data.get(city, f"No weather data for {city}")

def get_joke():
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "A SQL query walks into a bar and asks: 'Can I join you?'",
        "There are only 10 kinds of people: those who understand binary and those who don’t."
    ]
    return random.choice(jokes)

# 4. Agent logic
def run_agent(user_input):
    # Step 1: Ask the model if a tool is needed
    system_prompt = (
        "You are an AI agent that can use the following tools via MCP:\n"
        + json.dumps(TOOLS, indent=2)
        + "\nIf a tool is needed, respond ONLY in JSON as:\n"
        + '{"tool": "tool_name", "arguments": {...}}\n'
        + "Otherwise, answer directly.\n"
        + f"User: {user_input}"
    )

    response = client.models.generate_content(model=MODEL, contents=system_prompt)
    model_text = response.text.strip()
    print(f"\nModel raw output:\n{model_text}\n")

    # Step 2: Parse tool decision
    try:
        decision = json.loads(model_text) # Load the json output (from the model) into a dictionary
        tool = decision["tool"]
        args = decision.get("arguments", {})
    except Exception:
        # Direct answer (no tool call)
        print("AI:", model_text)
        return

    # Step 3: Execute tool locally
    print(f"Model requested tool: {tool}({args})")

    if tool == "get_weather":
        result = get_weather(args.get("city", ""))
    elif tool == "get_joke":
        result = get_joke()
    else:
        result = f"Unknown tool: {tool}"

    # Step 4: Ask Gemini to respond naturally
    followup = (
        f"The user asked: '{user_input}'. "
        f"The tool '{tool}' returned this result: '{result}'. "
        "Now, answer the user naturally using this information."
    )

    final_prompt = (
        "You are now responding directly to the user. "
        "Do not use JSON or structured formatting. "
        "Respond conversationally and clearly.\n\n"
        + followup
    )

    final = client.models.generate_content(model=MODEL, contents=final_prompt)
    print("AI:", final.text.strip())

# 5. Main loop
if __name__ == "__main__":
    print("MCP + Gemini 2.5 Demo\nType 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_agent(user_input)
