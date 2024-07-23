import torch
from dataclasses import dataclass
from typing import Optional, Mapping, OrderedDict
from transformers.onnx import OnnxConfig
from transformers.utils import ModelOutput
from transformers import BertModel


class SBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([
            ("input_ids", {0: "batch", 1: "sequence"}),
            ("attention_mask", {0: "batch", 1: "sequence"})
        ])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([
            ("last_hidden_state", {0: "batch", 1: "sequence"})
        ])


@dataclass
class EmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


class CustomEmbeddingBertModel(BertModel):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        embeddings = super().forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds,
                                     output_attentions=True,
                                     output_hidden_states=True,
                                     return_dict=True)
        # Not sure about this, I copied it from https://github.com/UKPLab/sentence-transformers/issues/46#issuecomment-1651984758
        mean_embedding = embeddings.last_hidden_state.mean(dim=1)
        return EmbeddingOutput(last_hidden_state=mean_embedding)
