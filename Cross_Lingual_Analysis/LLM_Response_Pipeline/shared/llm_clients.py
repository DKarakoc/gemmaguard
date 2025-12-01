"""LLM API clients for OpenAI and OpenRouter."""

import os
import time
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")


class OpenAIClient:
    """Client for OpenAI models (GPT-4o, GPT-4o-mini, GPT-5-nano, etc.)"""

    def __init__(self, model: str = "gpt-4o-2024-05-13", max_tokens: int = 500, temperature: float = 0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # GPT-5-nano uses responses.create instead of chat.completions.create)
        self.is_gpt5_nano = "gpt-5-nano" in model.lower()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 30
    ) -> Dict:
        """
        Generate response from OpenAI model.

        Returns:
            dict: {
                'response': str,
                'finish_reason': str,
                'prompt_tokens': int,
                'completion_tokens': int,
                'total_tokens': int,
                'error': Optional[str]
            }
        """
        try:
            if self.is_gpt5_nano:
                # GPT-5-nano uses responses.create API
                input_text = prompt
                if system_prompt:
                    input_text = f"{system_prompt}\n\n{prompt}"

                response = self.client.responses.create(
                    model=self.model,
                    input=input_text,
                    reasoning={
                        "effort": "minimal",
                        "summary": "detailed"
                    }
                )
                response_text = ""
                reasoning_summary = []

                for item in response.output:
                    if item.type == 'message':
                        # Extract message text
                        for content in item.content:
                            if hasattr(content, 'text'):
                                response_text = content.text
                                break

                    elif item.type == 'reasoning':
                        # Extract reasoning summary
                        if hasattr(item, 'summary') and item.summary is not None:
                            for summary_item in item.summary:
                                if hasattr(summary_item, 'text'):
                                    reasoning_summary.append({
                                        'text': summary_item.text,
                                        'type': summary_item.type if hasattr(summary_item, 'type') else 'unknown'
                                    })

                # Calculate cost
                input_cost = (response.usage.input_tokens / 1_000_000) * 0.15
                output_cost = (response.usage.output_tokens / 1_000_000) * 0.60
                total_cost = input_cost + output_cost

                return {
                    'response': response_text,
                    'finish_reason': response.status,
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'reasoning_summary': reasoning_summary,  # NEW: Extended thinking information
                    'estimated_cost': total_cost,
                    'error': None
                }

            else:
                # Standard chat.completions API for GPT-4o, GPT-4o-mini, etc.
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=timeout
                )

                choice = response.choices[0]
                message = choice.message

                return {
                    'response': message.content or "",
                    'finish_reason': choice.finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'error': None
                }

        except Exception as e:
            return {
                'response': "",
                'finish_reason': 'error',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'error': str(e)
            }


class OpenRouterClient:
    """Client for OpenRouter models (Mistral, Llama, etc.)"""

    PRICING = {
        # Mistral models (per 1M tokens)
        "mistralai/mistral-7b-instruct": {"input": 0.06, "output": 0.06},
        "mistralai/mistral-7b-instruct-v0.1": {"input": 0.06, "output": 0.06},
        "mistralai/mistral-7b-instruct-v0.2": {"input": 0.06, "output": 0.06},
        "mistralai/mistral-7b-instruct-v0.3": {"input": 0.06, "output": 0.06},
        "mistralai/mixtral-8x7b-instruct": {"input": 0.24, "output": 0.24},
        "mistralai/mixtral-8x22b-instruct": {"input": 0.65, "output": 0.65},

        # Llama models
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.80, "output": 0.80},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.10, "output": 0.10},
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.65, "output": 0.65},
        "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},

        # Qwen models
        "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.35},

        # DeepSeek models
        "deepseek/deepseek-chat-v3-0324": {"input": 0.40, "output": 1.75},
        "deepseek/deepseek-chat-v3-0324:free": {"input": 0.0, "output": 0.0},
        "deepseek/deepseek-chat-v3.1": {"input": 0.20, "output": 0.80},
        "deepseek/deepseek-v3.2-exp": {"input": 0.27, "output": 0.40},
        "deepseek/deepseek-r1": {"input": 0.30, "output": 1.20},
        "deepseek/deepseek-r1:free": {"input": 0.0, "output": 0.0},

        # Claude models
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},

        # Gemini models
        "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
        "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "google/gemini-2.0-flash": {"input": 0.0, "output": 0.0},
        "google/gemini-2.0-flash-exp:free": {"input": 0.0, "output": 0.0},
        "google/gemini-2.0-flash-thinking-exp:free": {"input": 0.0, "output": 0.0},
    }

    def __init__(
        self,
        model: str = "mistralai/mistral-7b-instruct-v0.1",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 30
    ) -> Dict:
        """
        Generate response from OpenRouter model.

        Returns:
            dict: {
                'response': str,
                'finish_reason': str,
                'prompt_tokens': int,
                'completion_tokens': int,
                'total_tokens': int,
                'estimated_cost': float,
                'error': Optional[str]
            }
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=timeout
            )

            choice = response.choices[0]
            message = choice.message

            # Calculate cost
            pricing = self.PRICING.get(self.model, {"input": 0.5, "output": 0.5})
            prompt_cost = response.usage.prompt_tokens / 1_000_000 * pricing["input"]
            completion_cost = response.usage.completion_tokens / 1_000_000 * pricing["output"]
            total_cost = prompt_cost + completion_cost

            return {
                'response': message.content or "",
                'finish_reason': choice.finish_reason,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'estimated_cost': total_cost,
                'error': None
            }

        except Exception as e:
            return {
                'response': "",
                'finish_reason': 'error',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'estimated_cost': 0.0,
                'error': str(e)
            }

    @classmethod
    def estimate_cost(cls, model: str, num_samples: int, avg_tokens_per_sample: int = 300) -> float:
        """Estimate total cost"""
        pricing = cls.PRICING.get(model, {"input": 0.5, "output": 0.5})
        avg_input = avg_tokens_per_sample * 0.4
        avg_output = avg_tokens_per_sample * 0.6

        input_cost = (avg_input * num_samples / 1_000_000) * pricing["input"]
        output_cost = (avg_output * num_samples / 1_000_000) * pricing["output"]

        return input_cost + output_cost
