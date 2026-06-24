"""
Real MCP Client — connects to mcp_sdk.py over the MCP protocol (stdio transport)
=================================================================================

How it works
------------
1. This client spawns mcp_sdk.py as a subprocess (stdio transport).
2. It performs the MCP handshake (initialize).
3. It calls list_tools() to discover what the server exposes.
4. It calls get_weather via call_tool() — a real MCP protocol call, not a direct
   Python function call. The server executes the tool and sends the result back
   as a structured MCP message.

This is the full MCP flow: client ↔ protocol ↔ server, with the tool living
entirely on the server side.
"""

import asyncio
import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_sdk.py")


async def main():
    # Spawn the MCP server as a subprocess communicating over stdin/stdout
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT, "server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── Step 1: MCP handshake ─────────────────────────────────────
            await session.initialize()
            print("Connected to MCP server.\n")

            # ── Step 2: Discover available tools ─────────────────────────
            tools_response = await session.list_tools()
            print(f"Tools exposed by the server ({len(tools_response.tools)} found):")
            for tool in tools_response.tools:
                print(f"  [{tool.name}]  {tool.description}")
                schema = tool.inputSchema or {}
                props = schema.get("properties", {})
                if props:
                    for param, meta in props.items():
                        print(f"    param: {param} ({meta.get('type', '?')})")
            print()

            # ── Step 3: Call get_weather for several cities ───────────────
            cities = ["Tel Aviv", "London", "Tokyo"]
            for city in cities:
                print(f"--- {city} ---")
                result = await session.call_tool("get_weather", {"city": city})

                for content in result.content:
                    try:
                        data = json.loads(content.text)
                        print(f"  Temperature : {data['temperature_c']} °C")
                        print(f"  Wind speed  : {data['windspeed_kmh']} km/h")
                        print(f"  Direction   : {data['winddirection_deg']}°")
                        print(f"  Time        : {data['time']}")
                    except (json.JSONDecodeError, KeyError):
                        print(" ", content.text)
                print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except* anyio.BrokenResourceError:
        # Expected on clean exit: the server closes its stdout after the last
        # tool call, causing the background stdout_reader task to fail during
        # teardown. All tool calls already completed successfully at this point.
        pass
