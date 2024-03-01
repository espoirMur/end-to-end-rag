from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
import torch
from pathlib import Path

model_path = Path.cwd().joinpath('models')
model_id = 'bio-gpt-qa'
model_path = model_path.joinpath(model_id)


set_seed(42)

tokenizer = BioGptTokenizer.from_pretrained(model_path,  local_files_only=True)
model = BioGptForCausalLM.from_pretrained(model_path, local_files_only=True)

input_ = f"'question:what is the cause of covid ? context: the cause of covid is a virus'"
encoded_input = tokenizer([input_],
                          return_tensors='pt',
                          max_length=1024,
                          truncation=True)


past_key_values = None
return_text = []
for i in range(10):
    output = model(**encoded_input,
                   past_key_values=past_key_values, output_attentions=True)
    past_key_values = output.past_key_values
    if past_key_values:
        print("the shape of the past key is", past_key_values[0][0].shape)
        print("the shape of the past value is", past_key_values[0][1].shape)
    token = torch.argmax(output.logits[..., -1, :])

    context = token.unsqueeze(0)

    return_text += [token.tolist()]


sequence = tokenizer.decode(return_text)
sequence = sequence.split(".")[:-1]
print(sequence)
