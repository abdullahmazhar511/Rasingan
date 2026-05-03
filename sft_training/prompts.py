def llama3_get_prompt(system_prompt, user_content):
    """Llama 3 Instruct Format"""
    return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def phi3_get_prompt(system_prompt, user_content):
    """Phi-3 Instruct Format"""
    return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_content}<|end|>\n<|assistant|>\n"

def mistral_get_prompt(system_prompt, user_content):
    """Mistral/Ministral Format"""
    return f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"

def qwen3_get_prompt(system_prompt, user_content):
    """Qwen 2.5/3 Format"""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def llama2_get_prompt(system_prompt, user_content):
    """Llama 2 Format"""
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"

def get_prompt_template(model_name):
    """
    Legacy helper kept for compatibility.
    """
    model_name = model_name.lower()
    if "qwen" in model_name:
        return "<|im_start|>system\nYou are a helpful mental health assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in model_name:
        return "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful mental health assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "phi" in model_name:
        return "<|system|>\nYou are a helpful mental health assistant.<|end|>\n<|user|>\n{instruction}\n{input}<|end|>\n<|assistant|>\n"
    elif "mistral" in model_name:
        return "[INST] {instruction}\n{input} [/INST]"
    else:
        return "System: You are a helpful assistant.\nUser: {instruction}\n{input}\nAssistant: "
