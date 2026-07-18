import torch

from validation_seeds import create_validation_seeds


def create_validation_generators(num_samples: int, base_seed: int):
    """Create reproducible per-index generators shared across prompt variants."""
    return [
        torch.Generator().manual_seed(seed)
        for seed in create_validation_seeds(num_samples, base_seed)
    ]


def _encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    max_sequence_length=512,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list
