
from torch import nn
from transformers import BioGptForCausalLM


class BioGptForCausalLMPrompt(BioGptForCausalLM):

    """
    Update model to handle the extras vocab tokens due to the prompt setup.
    Not 100 % sure how this works in general but it works, I will get back to this.
    """

    def __init__(self, config, shape_difference=0):
        # If the state dict has more token than the model embedding, we update the model embedding layer and the output projection layer to accommodate the new tokens
        super().__init__(config)
        if shape_difference > 0:
            self.biogpt.embed_tokens = nn.Embedding(
                config.vocab_size + shape_difference, self.biogpt.embed_dim, self.biogpt.padding_idx)
            self.output_projection = nn.Linear(
                config.hidden_size, config.vocab_size + shape_difference, bias=False)
