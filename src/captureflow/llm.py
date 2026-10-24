"""The one model call per chat turn: an OpenAI-compatible adapter and its prompt."""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .read_agent import RESPOND_TOOL, AgentFinalResponse, AgentModelTurn, AgentToolCall


class ModelError(RuntimeError):
    """A configured model could not supply a schema-valid response."""


class InvalidStructuredOutput(ModelError):
    """The model answered, but not with a decision that fits the schema and its inputs."""


class CaptureFlowModel(Protocol):
    def next_read_agent_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentModelTurn: ...


@dataclass(frozen=True)
class ModelSettings:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 30

    @classmethod
    def from_environment(cls) -> "ModelSettings":
        api_key = os.getenv("CAPTUREFLOW_API_KEY")
        model = os.getenv("CAPTUREFLOW_MODEL")
        if not api_key or not model:
            raise ModelError("Set CAPTUREFLOW_API_KEY and CAPTUREFLOW_MODEL before chatting.")
        try:
            timeout_seconds = float(os.getenv("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ModelError("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS must be numeric.") from error
        if timeout_seconds <= 0:
            raise ModelError("CAPTUREFLOW_MODEL_TIMEOUT_SECONDS must be positive.")
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.getenv("CAPTUREFLOW_BASE_URL"),
            timeout_seconds=timeout_seconds,
        )


Schema = TypeVar("Schema", bound=BaseModel)


class OpenAICompatibleModel:
    """A small adapter for providers exposing OpenAI-compatible chat completions."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def next_read_agent_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AgentModelTurn:
        """Normalize one OpenAI-compatible tool-calling response for the agent loop."""

        try:
            request: dict[str, Any] = {
                "model": self.settings.model,
                "timeout": self.settings.timeout_seconds,
                "messages": messages,
                "tools": tools,
            }
            if len(tools) == 1 and tools[0]["function"]["name"] == RESPOND_TOOL:
                request["tool_choice"] = {"type": "function", "function": {"name": RESPOND_TOOL}}
            if (
                self.settings.model.casefold().startswith("qwen")
                and self.settings.base_url is not None
                and "aliyuncs.com" in self.settings.base_url.casefold()
            ):
                # DashScope Qwen 3.x defaults to thinking mode. Its function-calling
                # guide recommends disabling thinking, which also avoids having to
                # replay provider-specific reasoning_content on every tool round.
                request["extra_body"] = {"enable_thinking": False}
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message
            raw_calls = getattr(message, "tool_calls", None) or []
            responses = [raw_call for raw_call in raw_calls if raw_call.function.name == RESPOND_TOOL]
            if responses:
                if len(raw_calls) != 1:
                    raise InvalidStructuredOutput(f"The model called {RESPOND_TOOL} together with other tools.")
                return AgentModelTurn(
                    final_response=AgentFinalResponse.model_validate_json(responses[0].function.arguments)
                )
            if raw_calls:
                calls: list[AgentToolCall] = []
                for raw_call in raw_calls:
                    arguments = json.loads(raw_call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    calls.append(
                        AgentToolCall(
                            id=raw_call.id,
                            name=raw_call.function.name,
                            arguments=arguments,
                        )
                    )
                return AgentModelTurn(tool_calls=calls)
            raise InvalidStructuredOutput(f"The model answered without calling {RESPOND_TOOL}.")
        except ValidationError as error:
            raise InvalidStructuredOutput(
                f"The model returned invalid agent output: {describe_validation_error(error)}"
            ) from error
        except (IndexError, ValueError, json.JSONDecodeError) as error:
            raise InvalidStructuredOutput("The model returned invalid agent tool arguments.") from error
        except Exception as error:
            if isinstance(error, ModelError):
                raise
            raise ModelError(f"Model request failed: {error}") from error

    def _complete(self, schema: type[Schema], system: str, payload: dict[str, Any]) -> Schema:
        prompt = json.dumps(
            {"schema": schema.model_json_schema(), "input": payload}, ensure_ascii=False
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                timeout=self.settings.timeout_seconds,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ModelError("The model returned an empty response.")
            return schema.model_validate_json(content)
        except ValidationError as error:
            raise InvalidStructuredOutput(
                f"The model returned invalid structured output: {describe_validation_error(error)}"
            ) from error
        except (IndexError, ValueError) as error:
            raise ModelError(f"The model returned invalid structured output: {error}") from error
        except Exception as error:
            if isinstance(error, ModelError):
                raise
            raise ModelError(f"Model request failed: {error}") from error


def describe_validation_error(error: ValidationError) -> str:
    """Name each failing field and rule without echoing the model's values back."""

    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc']) or '<root>'}: {detail['msg']}"
        for detail in error.errors(include_input=False, include_url=False)
    )
