import os
import requests
import json
from dotenv import load_dotenv
from google import genai
from mcp.server.fastmcp import FastMCP

load_dotenv()

server = FastMCP("weather-mcp-server")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Gemini client (used by the agent loop below)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"


# -------------------------------------------------------------------
# Tool registered with the MCP server via @server.tool()
# FastMCP exposes this tool's name, description, and parameter schema
# to any MCP client that connects. The function is also callable
# directly, which is what the agent loop does.
# -------------------------------------------------------------------
@server.tool(
    name="get_weather",
    description="Get current weather for a city by name using Open-Meteo."
)
def get_weather(city: str) -> dict:
    """Fetches current weather data from the Open-Meteo API for a given city."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or not geo_resp["results"]:
        return {"error": f"Could not find coordinates for {city}."}

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]
    country = geo_resp["results"][0].get("country", "Unknown")

    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    weather_resp = requests.get(OPEN_METEO_URL, params=params).json()

    if "current_weather" not in weather_resp:
        return {"error": f"No weather data available for {city}."}

    current = weather_resp["current_weather"]
    return {
        "city": city,
        "country": country,
        "temperature_c": current["temperature"],
        "windspeed_kmh": current["windspeed"],
        "winddirection_deg": current["winddirection"],
        "time": current["time"]
    }


# Tool registry shown to Gemini — mirrors what FastMCP exposes to MCP clients
TOOLS = {
    "get_weather": {
        "description": "Get current weather for a city by name using Open-Meteo.",
        "parameters": {"city": "string"}
    }
}


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

    # Step 2: Parse the tool call
    try:
        decision = json.loads(model_text)
        tool = decision["tool"]
        args = decision.get("arguments", {})
    except Exception:
        print("AI:", model_text)
        return

    # Step 3: Execute the FastMCP-registered tool directly
    print(f"Model requested tool: {tool}({args})")

    if tool == "get_weather":
        result = get_weather(args.get("city", ""))
    else:
        result = {"error": f"Unknown tool: {tool}"}

    # Step 4: Feed result back to Gemini for a natural reply
    followup = (
        f"The user asked: '{user_input}'. "
        f"The tool '{tool}' returned: {json.dumps(result)}. "
        "Respond to the user in a friendly, natural way. Do not use JSON."
    )

    final = client.models.generate_content(model=MODEL, contents=followup)
    print("AI:", final.text.strip())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Start the FastMCP server (for real MCP clients)
        server.run()
    else:
        # Run the Gemini agent chat loop
        print("MCP SDK + Gemini Agent Weather Demo\nType 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            run_agent(user_input)
