

from mcp.server.fastmcp import FastMCP

mcp = FastMCP()
mcp.settings.port=8001

@mcp.tool()
def sub(a: int, b: int) -> int:
    """Subtracts two integers."""
    print(f"[MCP Server] Received add request: {a} + {b}")
    result = a - b
    print(f"[MCP Server] Returning result: {result}")
    return result

if __name__ == "__main__":
    print("Starting minimal MCP server on http://localhost:8001/sse")
    
    mcp.run(transport='sse')

