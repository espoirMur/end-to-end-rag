
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
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            encoded_inputs: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY, padding=True, truncation=True
            )
            # tensorrt uses int32 as input type, ort uses int64
            encoded_inputs = {k: v.astype(np.int32)
                              for k, v in encoded_inputs.items()}
            # communicate the tokenization results to Triton server
            outputs = list()
            input_ids = encoded_inputs["input_ids"]
            tensor_input_ids = pb_utils.Tensor("input_ids", input_ids)
            tensor_max_length = pb_utils.Tensor(
                "max_length", np.array([512], dtype=np.int32))
            tensor_min_length = pb_utils.Tensor(
                "min_length", np.array([128], dtype=np.int32))
            tensor_num_beams = pb_utils.Tensor(
                "num_beams", np.array([5], dtype=np.int32))
            tensor_num_return_sequences = pb_utils.Tensor(
                "num_return_sequences", np.array([1], dtype=np.int32))
            tensor_length_penalty = pb_utils.Tensor(
                "length_penalty", np.array([1.0], dtype=np.float32))
            tensor_repetition_penalty = pb_utils.Tensor(
                "repetition_penalty", np.array([1.4], dtype=np.float32))

            outputs.append(tensor_input_ids)
            outputs.append(tensor_max_length)
            outputs.append(tensor_min_length)
            outputs.append(tensor_num_beams)
            outputs.append(tensor_num_return_sequences)
            outputs.append(tensor_length_penalty)
            outputs.append(tensor_repetition_penalty)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=outputs)
            responses.append(inference_response)

        return responses
