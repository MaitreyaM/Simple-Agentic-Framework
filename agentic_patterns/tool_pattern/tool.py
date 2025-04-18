# --- START OF MODIFIED tool.py ---

import json
from typing import Callable, Any # Added Any

def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature (schema) for a given function in JSON Schema format.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary representing the function's schema.
    """
    # Map Python types to JSON Schema types
    type_mapping = {
        "int": "integer",
        "str": "string",
        "bool": "boolean",
        "float": "number",
        "list": "array", # Basic support for lists
        "dict": "object", # Basic support for dicts
    }

    parameters = {"type": "object", "properties": {}, "required": []}
    for name, type_hint in fn.__annotations__.items():
        if name == "return":
            continue
        param_type_name = getattr(type_hint, "__name__", str(type_hint)) # Handle complex types slightly better
        schema_type = type_mapping.get(param_type_name, "string") # Default to string if type unknown

        parameters["properties"][name] = {"type": schema_type}
        # Add description if available in docstring (requires parsing docstring, omitted for simplicity here)
        # parameters["properties"][name]["description"] = "..."

        # Assume all parameters are required for now
        parameters["required"].append(name)

    # Final tool schema structure expected by OpenAI/Groq API
    fn_schema: dict = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": fn.__doc__,
            "parameters": parameters,
        }
    }

    return fn_schema


def validate_arguments(tool_call_args: dict, tool_schema: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary based on the tool's JSON schema.
    NOTE: This is a simplified validator. For production, use a robust JSON Schema validator.

    Args:
        tool_call_args (dict): The arguments provided for the tool call (usually strings from LLM).
        tool_schema (dict): The JSON schema for the tool's parameters.

    Returns:
        dict: The arguments dictionary with values converted to the correct types if possible.

    Raises:
        ValueError: If conversion fails for a required argument.
    """
    properties = tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
    validated_args = {}

    type_mapping = {
        "integer": int,
        "string": str,
        "boolean": bool,
        "number": float,
        "array": list, # Basic conversion
        "object": dict # Basic conversion
    }

    for arg_name, arg_value in tool_call_args.items():
        prop_schema = properties.get(arg_name)
        if not prop_schema:
            # Argument not defined in schema, potentially skip or warn
            print(f"Warning: Argument '{arg_name}' not found in tool schema.")
            validated_args[arg_name] = arg_value # Pass through unknown args for now
            continue

        expected_type_name = prop_schema.get("type")
        expected_type = type_mapping.get(expected_type_name)

        if expected_type:
            try:
                # Attempt conversion only if the type is not already correct
                if not isinstance(arg_value, expected_type):
                    # Special handling for boolean strings
                    if expected_type is bool and isinstance(arg_value, str):
                         if arg_value.lower() in ['true', '1', 'yes']:
                             validated_args[arg_name] = True
                         elif arg_value.lower() in ['false', '0', 'no']:
                              validated_args[arg_name] = False
                         else:
                              raise ValueError(f"Cannot convert string '{arg_value}' to boolean.")
                    # Basic handling for array/object assuming JSON string
                    elif expected_type in [list, dict] and isinstance(arg_value, str):
                        try:
                            validated_args[arg_name] = json.loads(arg_value)
                            if not isinstance(validated_args[arg_name], expected_type):
                                raise ValueError(f"Decoded JSON for '{arg_name}' is not the expected type '{expected_type_name}'.")
                        except json.JSONDecodeError:
                             raise ValueError(f"Argument '{arg_name}' with value '{arg_value}' is not valid JSON for type '{expected_type_name}'.")
                    else:
                        validated_args[arg_name] = expected_type(arg_value)
                else:
                    # Type is already correct
                    validated_args[arg_name] = arg_value
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting argument '{arg_name}' with value '{arg_value}' to type '{expected_type_name}': {e}")
        else:
             # Unknown type in schema, pass through
            validated_args[arg_name] = arg_value

    # Check for missing required arguments (optional, depends on strictness)
    # required_args = tool_schema.get("function", {}).get("parameters", {}).get("required", [])
    # for req_arg in required_args:
    #     if req_arg not in validated_args:
    #         raise ValueError(f"Missing required argument: '{req_arg}'")


    return validated_args


class Tool:
    """
    A class representing a tool that wraps a callable and its schema.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_schema (dict): Dictionary representing the function's schema in JSON Schema format.
        fn_signature (str): JSON string representation of the function's signature (legacy, kept for potential compatibility).
    """

    def __init__(self, name: str, fn: Callable, fn_schema: dict): # Added fn_schema
        self.name = name
        self.fn = fn
        self.fn_schema = fn_schema # Store the schema dictionary
        # Keep the string version for now if needed elsewhere, though ideally phase out
        self.fn_signature = json.dumps(fn_schema)

    def __str__(self):
        # Return the schema string representation
        return json.dumps(self.fn_schema, indent=2)

    def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        # Consider adding validation here using self.fn_schema before calling self.fn
        try:
            validated_kwargs = validate_arguments(kwargs, self.fn_schema)
            return self.fn(**validated_kwargs)
        except ValueError as e:
             # Handle validation errors appropriately
             print(f"Argument validation failed for tool {self.name}: {e}")
             # Return an error message or raise the exception depending on desired behavior
             return f"Error: {e}"
        except Exception as e:
             print(f"Error executing tool {self.name}: {e}")
             return f"Error executing tool: {e}"


def tool(fn: Callable):
    """
    A decorator that wraps a function into a Tool object, including its JSON schema.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its schema.
    """

    def wrapper():
        fn_schema = get_fn_signature(fn) # Get the schema dictionary
        # Ensure the schema is valid before creating the Tool object
        if not fn_schema or 'function' not in fn_schema or 'name' not in fn_schema['function']:
             raise ValueError(f"Could not generate valid schema for function {fn.__name__}")
        return Tool(
            name=fn_schema["function"]["name"], # Get name from schema
            fn=fn,
            fn_schema=fn_schema # Pass the schema dictionary
        )

    return wrapper()

# --- END OF MODIFIED tool.py ---