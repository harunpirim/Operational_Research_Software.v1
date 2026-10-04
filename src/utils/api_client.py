"""
API Client Utility - Unified interface for Anthropic and OpenAI
"""

import os
from typing import Optional, Dict, Any
from enum import Enum


class APIProvider(str, Enum):
    """Supported API providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class APIClient:
    """
    Unified API client that supports both Anthropic Claude and OpenAI models.
    Automatically detects which API key is available or uses the configured provider.
    """

    # Claude model-ID prefixes that still accept `temperature`
    # (Opus 4.7+, Sonnet 5.x and later models reject sampling parameters).
    _ANTHROPIC_SAMPLING_MODELS = (
        "claude-3",
        "claude-sonnet-4-",
        "claude-opus-4-0",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    )

    # Models that take server-side refusal fallback (fallbacks="default").
    _ANTHROPIC_FALLBACK_MODELS = {
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-fable-5-1",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the API client.

        Args:
            provider: 'anthropic' or 'openai'. If None, auto-detects from available keys.
            api_key: API key. If None, reads from environment variables.
            model: Model name. If None, uses default from environment or provider default.
        """
        self.provider = self._determine_provider(provider, api_key)
        self.api_key = self._get_api_key(api_key)
        self.model = model or self._get_default_model()
        self.client = self._create_client()

    def _determine_provider(
        self, provider: Optional[str], api_key: Optional[str]
    ) -> APIProvider:
        """Determine which provider to use."""
        if provider:
            try:
                return APIProvider(provider.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown provider: {provider}. Use 'anthropic' or 'openai'"
                )

        # Auto-detect from environment
        env_provider = os.getenv("AI_PROVIDER", "").lower()
        if env_provider in ["anthropic", "openai"]:
            return APIProvider(env_provider)

        # Auto-detect from available API keys
        if api_key:
            # If key starts with sk- it's likely OpenAI, if it's a long string it might be Anthropic
            # But we can't reliably detect from key format, so check env vars
            pass

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if anthropic_key and not openai_key:
            return APIProvider.ANTHROPIC
        elif openai_key and not anthropic_key:
            return APIProvider.OPENAI
        elif anthropic_key and openai_key:
            # Both available, prefer Anthropic by default (backward compatibility)
            return APIProvider.ANTHROPIC
        else:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                "or pass api_key parameter."
            )

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """Get API key from parameter or environment."""
        if api_key:
            return api_key

        if self.provider == APIProvider.ANTHROPIC:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            return key
        else:  # OPENAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            return key

    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        env_model = os.getenv("DEFAULT_MODEL")
        if env_model:
            return env_model

        if self.provider == APIProvider.ANTHROPIC:
            return "claude-sonnet-5-5"
        else:  # OPENAI
            return "gpt-4o"

    def _create_client(self):
        """Create the appropriate client instance."""
        if self.provider == APIProvider.ANTHROPIC:
            from anthropic import Anthropic

            return Anthropic(api_key=self.api_key)
        else:  # OPENAI
            from openai import OpenAI

            return OpenAI(api_key=self.api_key)

    def create_message(
        self,
        messages: list,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        model: Optional[str] = None,
        effort: str = "low",
    ) -> Dict[str, Any]:
        """
        Create a chat completion message.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (OpenAI and older Claude models)
            model: Override default model
            effort: Thinking effort for current Claude models
                ('low' suits classification/extraction; 'medium' for code generation)

        Returns:
            Response dictionary with 'content' field containing the text

        Raises:
            RuntimeError: If a Claude model declines the request (stop_reason "refusal").
        """
        model = model or self.model

        if self.provider == APIProvider.ANTHROPIC:
            if model.startswith(self._ANTHROPIC_SAMPLING_MODELS):
                # anthropic SDK 1.x removed `temperature` from messages.create();
                # these older models still honour it via the request body.
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    extra_body={"temperature": temperature},
                )
            else:
                # Current models reject sampling params and always think; thinking
                # counts toward max_tokens, so leave room for it plus the reply.
                kwargs = dict(
                    model=model,
                    max_tokens=max(max_tokens, 16000),
                    messages=messages,
                    output_config={"effort": effort},
                )
                if model in self._ANTHROPIC_FALLBACK_MODELS:
                    # On a policy decline, the API re-runs the request on a
                    # fallback model chosen by the refusal category.
                    response = self.client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                        **kwargs,
                    )
                else:
                    response = self.client.messages.create(**kwargs)

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                raise RuntimeError(
                    f"{model} declined the request"
                    + (f" (category: {category})" if category else "")
                    + ". Try rephrasing the problem description."
                )
            return {
                # The first block may be a thinking block on newer models.
                "content": "".join(
                    b.text for b in response.content if b.type == "text"
                ),
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
        else:  # OPENAI
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
            }

    def get_provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.provider.value

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.model

    def test_connection(self) -> bool:
        """
        Test the API connection and print solver availability.

        Checks:
          1. AI API reachability (Anthropic / OpenAI).
          2. CVXPY solver availability (via ``verify_cvxpy_solvers``).

        Returns True if the AI API responds, False otherwise.
        """
        ok = False
        try:
            response = self.create_message(
                messages=[{"role": "user", "content": "Reply OK"}],
                max_tokens=10,
                temperature=0,
            )
            print(f"Connection successful! Model: {response['model']}")
            print(f"Response: {response['content']}")
            ok = True
        except Exception as e:
            print(f"Connection failed: {e}")

        try:
            from src.utils.solver_utils import verify_cvxpy_solvers

            cvxpy_info = verify_cvxpy_solvers()
            print(f"\nCVXPY available : {cvxpy_info['available']}")
            if cvxpy_info["available"]:
                print(f"CVXPY solvers   : {', '.join(cvxpy_info['solvers'])}")
                print(f"CVXPY default   : {cvxpy_info['default']}")
            elif cvxpy_info["error"]:
                print(f"CVXPY error     : {cvxpy_info['error']}")
        except Exception as e:
            print(f"\nCVXPY check skipped: {e}")

        return ok


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = APIClient()
    print(f"Provider: {client.get_provider_name()}")
    print(f"Model: {client.get_model_name()}")
    print(f"API Key: {client.api_key[:20]}...{client.api_key[-6:]}")
    print("Testing connection...")
    client.test_connection()
