# Copyright (c) 2026 The University of Texas at Austin and B-Tree Labs
# SPDX-License-Identifier: Apache-2.0

"""A LangChain chat model whose every call goes through the Axiom gateway.

The platform already hosts several foreign runtimes this way. From
``axiom.llm.anthropic_ingress``:

    Every request is **translated through the Axiom gateway**, so all clients
    ride the same routing tiers, fail-closed export-controlled enforcement,
    vault, and audit as ``axi chat`` — NOT a LiteLLM-style passthrough that
    would bypass the chokepoint.

This module is the same idea for LangChain. A graph built on
:class:`AxiomChatModel` cannot reach a provider except through the gateway, so
tier routing, export-control enforcement, credential handling and audit apply to
it exactly as they do to a native call. LangGraph becomes a client of the
platform rather than a second brain beside it.

**It fails closed.** The gateway degrades gracefully for its own callers: when
no provider is usable it returns ``success=False`` with placeholder text so a
CLI can carry on. Passing that back to a graph as an ordinary answer would
launder a failure into content — the graph would branch on a sentence that no
model produced. So a non-success response raises
:class:`AxiomGatewayUnavailable` here.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

__all__ = ["ROUTING_TIER_ENV", "AxiomChatModel", "AxiomGatewayUnavailable"]

#: Pin every call from this process to a routing tier. Mirrors
#: ``AXIOM_BRIDGE_ROUTING_TIER`` on the Anthropic/OpenAI ingress, so an
#: export-controlled deployment configures both bridges the same way.
ROUTING_TIER_ENV = "AXIOM_LANGGRAPH_ROUTING_TIER"

_VALID_TIERS = ("public", "export_controlled", "any")


class AxiomGatewayUnavailable(RuntimeError):
    """The gateway could not serve the request.

    Raised rather than returned so a graph halts instead of branching on
    placeholder text. The gateway's graceful-degradation contract is right for
    a CLI and wrong for an autonomous run.
    """


def _to_gateway_messages(messages: Sequence[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Split LangChain messages into a system prompt and the gateway's history.

    The gateway takes ``system`` separately, so system messages are lifted out
    and joined rather than left in the turn list where they would be sent as
    user content.
    """
    system_parts: list[str] = []
    history: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(str(message.content))
        elif isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            history.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            )
        elif isinstance(message, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "input": call.get("args", {}),
                    }
                    for call in message.tool_calls
                ]
            history.append(entry)
        else:
            # Unknown subclass: carry it as user content rather than dropping
            # it. Silently discarding a turn changes what the model saw.
            history.append({"role": "user", "content": str(message.content)})

    return "\n\n".join(p for p in system_parts if p), history


def _to_ai_message(response: Any) -> AIMessage:
    """Convert the gateway's CompletionResponse into an AIMessage."""
    tool_calls = [
        {
            "name": block.name,
            "args": dict(block.input or {}),
            "id": block.tool_id,
            "type": "tool_call",
        }
        for block in (response.tool_use or [])
    ]

    return AIMessage(
        content=response.text or "",
        tool_calls=tool_calls,
        response_metadata={
            "provider": response.provider,
            "model": response.model,
            "stop_reason": response.stop_reason,
        },
        usage_metadata={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
            "input_token_details": {"cache_read": response.cache_read_tokens},
        },
    )


class AxiomChatModel(BaseChatModel):
    """A LangChain chat model backed by the Axiom gateway.

    Args:
        routing_tier: ``"public"``, ``"export_controlled"`` or ``"any"``. An
            export-controlled deployment pins this, or sets
            ``AXIOM_LANGGRAPH_ROUTING_TIER``, so no graph in the process can
            reach a non-authorised provider. An explicit argument wins over the
            environment.
        task: task label the gateway routes on.
        max_tokens: generation cap per call.
        prefer: optional provider name to prefer, subject to tier.
        gateway_factory: override for tests. Defaults to constructing
            ``axiom.llm.gateway.Gateway`` on first use.
    """

    routing_tier: str = Field(default="")
    task: str = Field(default="chat")
    max_tokens: int = Field(default=4096)
    prefer: str | None = Field(default=None)
    gateway_factory: Callable[[], Any] | None = Field(default=None, exclude=True)

    _tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _gateway: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        tier = self.routing_tier or os.environ.get(ROUTING_TIER_ENV, "").strip() or "any"
        if tier not in _VALID_TIERS:
            raise ValueError(
                f"routing_tier must be one of {_VALID_TIERS}, got {tier!r}. "
                "An unrecognised tier is refused rather than defaulted, because "
                "defaulting a mistyped 'export_controlled' to 'any' would send "
                "controlled work to a public provider."
            )
        object.__setattr__(self, "routing_tier", tier)

    @property
    def _llm_type(self) -> str:
        return "axiom-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"routing_tier": self.routing_tier, "task": self.task}

    def _get_gateway(self) -> Any:
        if self._gateway is None:
            if self.gateway_factory is not None:
                self._gateway = self.gateway_factory()
            else:
                from axiom.llm.gateway import Gateway

                self._gateway = Gateway()
        return self._gateway

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> AxiomChatModel:
        """Bind tools, converted to the function-calling shape the gateway takes.

        Returns a copy so binding does not mutate a model another graph node
        holds.
        """
        bound = self.model_copy()
        # Assign privates normally. Pydantic keeps them in ``__pydantic_private__``,
        # so ``object.__setattr__`` would put a second copy in ``__dict__`` that
        # shadows it on read while later assignments went to the other one — the
        # lazily-built gateway would be written in one place and read from the
        # other, and ``_get_gateway`` would quietly return None.
        bound._gateway = self._gateway
        bound._tools = [convert_to_openai_tool(t) for t in tools]
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, history = _to_gateway_messages(messages)

        response = self._get_gateway().complete_with_tools(
            messages=history,
            system=system,
            tools=self._tools or None,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            task=kwargs.get("task", self.task),
            routing_tier=kwargs.get("routing_tier", self.routing_tier),
            prefer=kwargs.get("prefer", self.prefer),
        )

        if not getattr(response, "success", False):
            raise AxiomGatewayUnavailable(
                f"Axiom gateway did not serve the request "
                f"(provider={response.provider!r}, tier={self.routing_tier!r}): "
                f"{response.error or response.text}"
            )

        return ChatResult(generations=[ChatGeneration(message=_to_ai_message(response))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Not implemented.

        The gateway exposes ``stream_with_tools``, so this is a real gap rather
        than an impossibility. Raising is deliberate: LangChain's default
        fallback would silently call ``_generate`` and yield one chunk, which
        looks like streaming works and hides that nothing incremental happens.
        """
        raise NotImplementedError(
            "streaming is not wired yet; the gateway's stream_with_tools is the "
            "seam to build it on. Use .invoke() rather than .stream()."
        )


def bind_axiom_tools(model: AxiomChatModel, tools: Sequence[BaseTool]) -> AxiomChatModel:
    """Convenience wrapper mirroring ``model.bind_tools`` for readability."""
    return model.bind_tools(tools)
