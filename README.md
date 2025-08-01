# PDTrainer

Fine-tune your LLM using minimal data (down to 1 example) and computing power with the Prompt Distillation trainer.

## Usage

### Installation

Clone this repository, and then:

```bash
pip install "trl[peft]"
pip install flash-attn  # Recommended
pip install -e ./pd-trainer
```

### Dataset format

The `PDTrainer` supports both conversational and instruction dataset formats.

```json
[
  {
    "prompt": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Replace the background color in App.tsx with #f5f5f5."}
    ],
    "teacher_prompt": [
      {"role": "system", "content": "You are a helpful assistant.\n\nWhen modifying a code file, you need to first review the contents of the code file before deciding how to modify it."},
      {"role": "user", "content": "Replace the background color in App.tsx with #f5f5f5."}
    ],
    "completion": [{"role": "assistant", "content": "<Output generated using teacher_prompt>"}],
    "tools": []
  }
]
```

The `tools` field is optional, and its format is identical to [the dataset format used by SFTTrainer](https://huggingface.co/docs/trl/main/en/dataset_formats#tool-calling) in TRL.

**WARNING**: The `completion` field **MUST** be generated using the model to fine-tune and `teacher_prompt` (For OpenAI Python SDK, `[completion.choices[0].message.model_dump()]` is recommended). Using other models or directly modifying the `completion` field is pointless.

### CLI

Fine-tune using CLI:

```bash
pd-trainer --model_name_or_path qwen/Qwen3-32B --dataset ./path/to/dataset.json --save_dir ./outputs --attn_implementation flash_attention_2
```

## References

Kalle Kujanpää, Harri Valpola & Alexander Ilin (2024).  
**Knowledge Injection via Prompt Distillation**.  
arXiv preprint arXiv:2412.14964.  
<https://arxiv.org/abs/2412.14964>
