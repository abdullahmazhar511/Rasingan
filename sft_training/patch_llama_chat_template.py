import argparse
from transformers import AutoTokenizer


def make_trl_assistant_template(original_template: str) -> str:
    """Patch Meta's original template minimally for TRL assistant_only_loss.

    We only change the plain text assistant rendering branch to wrap assistant
    content in `{% generation %}` markers and keep all other original logic.
    """
    # Llama-style template
    llama_old_block = (
        "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
        "        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}"
    )

    llama_new_block = (
        "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
        "        {%- if message['role'] == 'assistant' %}\n"
        "            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
        "            {% generation %}{{- message['content'] | trim }}{% endgeneration %}\n"
        "            {{- '<|eot_id|>' }}\n"
        "        {%- else %}\n"
        "            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}\n"
        "        {%- endif %}"
    )

    # Qwen-style template assistant branch
    qwen_old_block = (
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content }}"
    )

    qwen_new_block = (
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' }}\n"
        "        {% generation %}{{- content }}{% endgeneration %}"
    )

    if llama_old_block in original_template:
        return original_template.replace(llama_old_block, llama_new_block)

    if qwen_old_block in original_template:
        return original_template.replace(qwen_old_block, qwen_new_block)

    raise ValueError("Could not locate a supported assistant rendering block in chat template")


def patch_tokenizer_chat_template(tokenizer):
    tokenizer.chat_template = make_trl_assistant_template(tokenizer.chat_template)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Patch tokenizer chat template for TRL assistant_only_loss")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    patch_tokenizer_chat_template(tok)
    tok.save_pretrained(args.output_dir)
    print(f"Saved patched tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
