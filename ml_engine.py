"""Model engine: loads the LLM once and serves generations.

Inspired by Aman Kharwal's tutorial:
https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/
"""

from __future__ import annotations

import os

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


class LLMEngine:
    """Loads the model once and keeps it in memory for the app's lifetime."""

    def __init__(self) -> None:
        self.pipe = None

    def load_model(self) -> None:
        """Load the model. Call once, on startup."""
        from transformers import pipeline

        print(f"Loading {MODEL_ID} ...")
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype="auto",  # let Transformers pick the right dtype per device
            device_map="auto",
        )
        print("Model loaded.")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's own tokenizer."""
        if self.pipe is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return len(self.pipe.tokenizer.encode(text))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Generate a completion for a prompt."""
        if self.pipe is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            formatted,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_full_text=False,  # return only the completion, no manual slicing
        )
        return outputs[0]["generated_text"]


llm_engine = LLMEngine()
