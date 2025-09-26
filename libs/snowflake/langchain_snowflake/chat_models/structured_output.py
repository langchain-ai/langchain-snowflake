"""Structured output functionality for Snowflake chat models."""

import json
import logging
import re

from langchain_core.messages import SystemMessage

from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeStructuredOutput:
    """Mixin class for Snowflake structured output functionality."""

    def with_structured_output(
        self,
        schema,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        **kwargs,
    ):
        """Use LLM intelligence to implement structured output.

        This method uses Snowflake Cortex Complete's reasoning capabilities
        to understand the desired output schema and format responses accordingly.

        Args:
            schema: Output schema (Pydantic model, TypedDict, or JSON schema)
            method: Method to use ("function_calling")
            include_raw: Whether to include raw response
            **kwargs: Additional keyword arguments
        """
        from langchain_core.utils.pydantic import is_basemodel_subclass

        # Import the base class from this module to avoid circular imports
        from .base import ChatSnowflake

        # Create a wrapper class that inherits from ChatSnowflake
        class StructuredChatSnowflake(ChatSnowflake):
            def __init__(self, base_model, target_schema, include_raw_output=False):
                # Extract only the known constructor parameters from the base model
                constructor_params = {}

                # Core parameters that ChatSnowflake constructor accepts
                known_params = [
                    "model",
                    "session",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "warehouse",
                    "database",
                    "schema",
                    "account",
                    "user",
                    "password",
                    "token",
                    "private_key_path",
                    "private_key_passphrase",
                ]

                for param in known_params:
                    if hasattr(base_model, param):
                        constructor_params[param] = getattr(base_model, param)

                # Call parent constructor with valid parameters
                super().__init__(**constructor_params)

                self._base_model = base_model
                self._target_schema = target_schema
                self._include_raw = include_raw_output

                # Set format-specific attributes
                if is_basemodel_subclass(target_schema):
                    self._ls_structured_output_format = "pydantic"
                    # Handle both Pydantic v1 and v2 compatibility
                    if hasattr(target_schema, "model_json_schema"):
                        # Pydantic v2
                        self._schema_dict = target_schema.model_json_schema()
                    elif hasattr(target_schema, "schema"):
                        # Pydantic v1
                        self._schema_dict = target_schema.schema()
                    else:
                        # Fallback for other formats
                        self._schema_dict = {"type": "object", "properties": {}}
                    self._schema_name = target_schema.__name__
                elif hasattr(target_schema, "__annotations__"):
                    self._ls_structured_output_format = "typeddict"
                    self._schema_dict = {"type": "object", "properties": {}}
                    self._schema_name = "TypedDict"
                else:
                    self._ls_structured_output_format = "json_schema"
                    self._schema_dict = target_schema
                    self._schema_name = "JSONSchema"

                # Copy important attributes from base model that might be private
                for attr in ["session", "_bound_tools", "_tool_choice"]:
                    if hasattr(base_model, attr):
                        setattr(self, attr, getattr(base_model, attr))

                # Set the format attribute on both the instance and as a class attribute for LangChain compatibility
                setattr(
                    self,
                    "ls_structured_output_format",
                    self._ls_structured_output_format,
                )

            @property
            def _ls_structured_output_format_dict(self):
                """Return structured output format metadata for LangChain callbacks."""
                return {
                    "schema": self._schema_dict,
                    "name": self._schema_name,
                    "format": self._ls_structured_output_format,
                }

            def dict(self, **kwargs):
                """Override dict to include structured output metadata."""
                result = super().dict(**kwargs)
                result["ls_structured_output_format"] = self._ls_structured_output_format_dict
                return result

            def invoke(self, input, config=None, **kwargs):
                """Override invoke to provide structured output using LLM intelligence."""
                # Enhance the input with intelligent schema instructions
                if isinstance(input, str):
                    enhanced_input = self._create_enhanced_prompt(input)
                else:
                    enhanced_input = self._add_schema_to_messages(input)

                # Get response from base model
                response = super().invoke(enhanced_input, config, **kwargs)

                # Use LLM intelligence to format the output
                formatted_output = self._format_output_intelligently(response.content)

                if self._include_raw:
                    return {"raw": response, "parsed": formatted_output}
                return formatted_output

            async def ainvoke(self, input, config=None, **kwargs):
                """Async version of structured invoke."""
                if isinstance(input, str):
                    enhanced_input = self._create_enhanced_prompt(input)
                else:
                    enhanced_input = self._add_schema_to_messages(input)

                response = await super().ainvoke(enhanced_input, config, **kwargs)
                formatted_output = self._format_output_intelligently(response.content)

                if self._include_raw:
                    return {"raw": response, "parsed": formatted_output}
                return formatted_output

            def _create_enhanced_prompt(self, user_input):
                """Create an enhanced prompt with schema instructions."""
                if self._ls_structured_output_format == "pydantic":
                    return f"""
Please provide your response in the exact format of this {self._schema_name} schema:

{json.dumps(self._schema_dict, indent=2)}

User Query: {user_input}

Respond with a valid JSON object that matches this schema exactly.
"""
                elif self._ls_structured_output_format == "typeddict":
                    keys = list(self._target_schema.__annotations__.keys())
                    return f"""
Provide your response as a JSON object with exactly these keys: {keys}

User Query: {user_input}

Respond with a valid JSON object containing all specified keys.
"""
                else:
                    return f"""
Provide your response according to this JSON schema:

{json.dumps(self._schema_dict, indent=2)}

User Query: {user_input}

Respond with a valid JSON object that conforms to the schema.
"""

            def _add_schema_to_messages(self, messages):
                """Add schema instruction to message list."""
                if self._ls_structured_output_format == "pydantic":
                    schema_instruction = SystemMessage(
                        content=f"""
Respond in the exact format of this {self._schema_name} schema:
{json.dumps(self._schema_dict, indent=2)}
Provide only a valid JSON object that matches this schema.
"""
                    )
                elif self._ls_structured_output_format == "typeddict":
                    keys = list(self._target_schema.__annotations__.keys())
                    schema_instruction = SystemMessage(
                        content=f"""
Respond as a JSON object with exactly these keys: {keys}
Provide only a valid JSON object containing all specified keys.
"""
                    )
                else:
                    schema_instruction = SystemMessage(
                        content=f"""
Respond according to this JSON schema:
{json.dumps(self._schema_dict, indent=2)}
Provide only a valid JSON object that conforms to the schema.
"""
                    )

                # Add system message if not present
                if not any(msg.type == "system" for msg in messages):
                    return [schema_instruction] + list(messages)
                return messages

            def _format_output_intelligently(self, response_content):
                """Use LLM intelligence to parse and format the output."""
                try:
                    # First, try to extract JSON from the response
                    # Clean the response - look for JSON patterns
                    json_content = response_content.strip()

                    # Remove markdown formatting if present
                    if "```json" in json_content:
                        json_content = json_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_content:
                        # Look for any code block
                        json_content = json_content.split("```")[1].split("```")[0].strip()

                    # Try to find JSON object pattern
                    json_match = re.search(r"\{.*\}", json_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group()

                    # Parse the JSON
                    parsed_data = json.loads(json_content)

                    # Create the appropriate output based on schema type
                    if self._ls_structured_output_format == "pydantic":
                        return self._target_schema(**parsed_data)
                    elif self._ls_structured_output_format == "typeddict":
                        return parsed_data
                    else:
                        return parsed_data

                except Exception as e:
                    # Log parsing failure and attempt LLM-assisted fallback
                    SnowflakeErrorHandler.log_error("parse structured output", e)
                    try:
                        format_prompt = f"""
Parse this response into the required format. Extract the relevant information and structure it correctly.

Target format: {self._ls_structured_output_format}
Schema: {json.dumps(self._schema_dict, indent=2) if hasattr(self, "_schema_dict") else "N/A"}

Response to parse: {response_content}

Return ONLY a valid JSON object that matches the required format.
"""
                        format_response = self._base_model.invoke(format_prompt)
                        json_content = format_response.content.strip()

                        if "```json" in json_content:
                            json_content = json_content.split("```json")[1].split("```")[0].strip()

                        parsed_data = json.loads(json_content)

                        if self._ls_structured_output_format == "pydantic":
                            return self._target_schema(**parsed_data)
                        else:
                            return parsed_data

                    except Exception as fallback_error:
                        # Log final fallback failure and return structured error response
                        SnowflakeErrorHandler.log_error("LLM-assisted parsing fallback", fallback_error)
                        return {
                            "error": f"Could not parse response into {self._ls_structured_output_format} format",
                            "raw": response_content,
                            "schema": (self._schema_dict if hasattr(self, "_schema_dict") else None),
                        }

        # Return the structured output wrapper
        return StructuredChatSnowflake(self, schema, include_raw)
