"""
MCP (Model Context Protocol) — Real Weather API Example
========================================================

This file demonstrates the core MCP principles using Gemini 2.5 Flash
and a real weather API (Open-Meteo), without a dedicated MCP server library.

MCP Principles Applied
----------------------
1. Tool Registry (TOOLS dict)
   MCP defines a standard way for models to discover available tools and their
   schemas. Here, TOOLS acts as that registry — each entry declares a tool name,
   description, and expected parameters, exactly what a real MCP server exposes.

2. Model-Driven Tool Selection
   The model is shown the tool registry and decides whether to call a tool or
   answer directly. This reflects MCP's core idea: the LLM is the orchestrator
   that picks which capability to invoke based on the user's intent.

3. Structured Tool Call (JSON)
   When the model decides a tool is needed it responds with a strict JSON schema:
     {"tool": "tool_name", "arguments": {...}}
   This mirrors MCP's standardized tool-call message format, making the contract
   between model and executor explicit and machine-readable.

4. Tool Execution Layer
   The agent parses the model's JSON decision, routes to the correct function
   (get_weather), and executes it. In a full MCP setup this routing happens inside
   the MCP server; here we implement it directly to show the underlying mechanics.

5. Context Injection (Two-Turn Pattern)
   After the tool runs, its result is injected back into a second prompt so the
   model can formulate a natural-language reply. This matches MCP's tool-result
   message that gets appended to the conversation context before the final response.

Flow
----
User input → Gemini decides tool → JSON parsed → real API called → result
injected → Gemini produces final answer
"""

import os
import requests
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

# 1. Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# 2. Define available tool
TOOLS = {
    "get_weather": {
        "description": "Get current weather for a city using Open-Meteo API",
        "parameters": {"city": "string"}
    }
}

# 3. Implement real API call
def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or not geo_resp["results"]:
        return f"Could not find coordinates for {city}."

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]
    country = geo_resp["results"][0].get("country", "Unknown")

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true"
    )
    weather_resp = requests.get(weather_url).json()

    if "current_weather" not in weather_resp:
        return f"No weather data available for {city}."

    current = weather_resp["current_weather"]
    temperature = current["temperature"]
    windspeed = current["windspeed"]

    return f"The current temperature in {city}, {country} is {temperature}°C with wind speed {windspeed} km/h."

# 4. Run the agent
def run_agent(user_input):
    # Step 1: Ask Gemini which tool to use
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

    # Step 2: Try parsing as JSON
    try:
        decision = json.loads(model_text)
        tool = decision["tool"]
        args = decision.get("arguments", {})
    except Exception:
        print("AI:", model_text)
        return

    # Step 3: Execute tool
    print(f"Model requested tool: {tool}({args})")

    if tool == "get_weather":
        result = get_weather(args.get("city", ""))
    else:
        result = f"Unknown tool: {tool}"

    # Step 4: Natural follow-up answer
    followup = (
        f"The user asked: '{user_input}'. "
        f"The tool '{tool}' returned this result: '{result}'. "
        "Now, respond to the user in a friendly, natural way. "
        "Do not use JSON. Just write text."
    )

    final = client.models.generate_content(model=MODEL, contents=followup)
    print("AI:", final.text.strip())


# 5. Main loop
if __name__ == "__main__":
    print("MCP + Gemini 2.5 Real Weather API Demo\nType 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_agent(user_input)
