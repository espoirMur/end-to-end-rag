
import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BioGptTokenizer, PreTrainedTokenizer, TensorType


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(
            args["model_repository"], args["model_version"])
        self.tokenizer = BioGptTokenizer.from_pretrained(path)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            sequences_output = pb_utils.get_input_tensor_by_name(
                request, "sequences")
            sequences = sequences_output.as_numpy().squeeze()
            text: List[np.ndarray] = self.tokenizer.decode(
                sequences, skip_special_tokens=True)
            text_array = np.array(text, dtype=object)
            response = pb_utils.Tensor(
                "sequences_text", text_array)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[response])
            responses.append(inference_response)
        return responses
