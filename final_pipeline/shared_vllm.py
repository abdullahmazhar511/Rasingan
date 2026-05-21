"""Process-wide shared vLLM client.

Patient (role-play) and Agent 2 (supervisor) both call into the same model,
so we load it once and reuse it.  Agent 1 stays on its own pipeline because
that's typically the model under evaluation.

Usage:
    from shared_vllm import get_shared_vllm
    llm = get_shared_vllm("Qwen/Qwen3-14B-Instruct")
    text = llm.chat([{"role": "system", ...}, {"role": "user", ...}])
"""

import threading
from typing import Dict, List, Optional

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

_SHARED: Dict[str, "SharedVLLM"] = {}
_LOCK = threading.Lock()


class SharedVLLM:
    """Thin wrapper around `vllm.LLM` with a chat() helper that uses the
    tokenizer's built-in chat template (no hardcoded special tokens)."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.55,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams  # heavy import — local

        self._SamplingParams = SamplingParams
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


def get_shared_vllm(model_name: str = DEFAULT_MODEL, **kwargs) -> SharedVLLM:
    """Return a process-wide shared client for `model_name`. First call loads
    the model; subsequent calls with the same name return the cached instance.
    Different model names produce independent clients (each its own GPU load)."""
    with _LOCK:
        if model_name not in _SHARED:
            _SHARED[model_name] = SharedVLLM(model_name=model_name, **kwargs)
        return _SHARED[model_name]


class RemoteVLLM:
    """HTTP-backed chat client for an external vLLM OpenAI-compatible server.

    Same `chat(messages, ...)` API as SharedVLLM, so it's a drop-in. Use this
    when the supervisor/patient model lives in a separate Python env (different
    vLLM version) and is hosted via `python -m vllm.entrypoints.openai.api_server`.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        from openai import OpenAI  # lazy
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            kwargs["stop"] = stop
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


def reset_shared_vllm(model_name: Optional[str] = None) -> None:
    """Drop cached client(s). Mostly for tests."""
    with _LOCK:
        if model_name is None:
            _SHARED.clear()
        else:
            _SHARED.pop(model_name, None)
