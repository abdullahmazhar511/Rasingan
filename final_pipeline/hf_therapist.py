"""In-process HuggingFace backend for Agent 1 (the therapist).

Unlike `SharedVLLM` / `RemoteVLLM`, this does NOT use vLLM in any form.
It loads the model directly with `transformers.AutoModelForCausalLM`,
runs generation via `model.generate(...)`, and exposes a `chat(messages, ...)`
interface that's signature-compatible with the vLLM wrappers — so
`Agent1_PrimaryTherapist` doesn't need to know which backend it has.

Lifecycle: construct ONE instance up-front (typically in run.py.main())
and pass it into every TherapyPipeline you create. The class is thread-safe
via an internal lock around `model.generate()`.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _pick_attn_impl() -> str:
    """Default to flash_attention_2 when the package is importable, else sdpa.
    Override with env var THERAPIST_ATTN_IMPL."""
    forced = os.environ.get("THERAPIST_ATTN_IMPL")
    if forced:
        return forced
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class HFTherapist:
    """In-process HuggingFace text-generation client with a vLLM-compatible chat() API."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda:0",
        dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_path:           HF repo id OR local dir (HF format with safetensors).
            adapter_path:         Optional PEFT/LoRA adapter dir to apply on top.
                                  If None or empty, the model is loaded standalone.
            device:               CUDA device string (e.g. "cuda:0") or "cpu".
            dtype:                Torch dtype; default = bf16 if supported, else fp16.
            attn_implementation:  HF attention backend; default auto-picks
                                  flash_attention_2 when installed, else sdpa.
            trust_remote_code:    HF flag (Qwen needs this for some configs).
        """
        self.model_path = model_path
        self.adapter_path = adapter_path or None
        self.device = device
        self.dtype = dtype or _pick_dtype()
        self.attn_implementation = attn_implementation or _pick_attn_impl()
        self._lock = threading.Lock()

        # Prefer the adapter's tokenizer if it ships one (sometimes adapters
        # introduce tokenizer changes); else use the model's.
        tok_source = self.adapter_path if (self.adapter_path and self._adapter_has_tokenizer()) else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
            attn_implementation=self.attn_implementation,
        )
        if self.adapter_path:
            from peft import PeftModel  # lazy import
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

    def _adapter_has_tokenizer(self) -> bool:
        if not self.adapter_path:
            return False
        return any(
            os.path.exists(os.path.join(self.adapter_path, f))
            for f in ("tokenizer.json", "tokenizer_config.json")
        )

    @torch.no_grad()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,  # accepted for API parity; ignored
    ) -> str:
        """Render messages with the tokenizer's chat template, generate, decode."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature is not None and temperature > 0.0),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with self._lock:
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
