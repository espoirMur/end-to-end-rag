## My Struggle delpoying model with gpt


I have been recently trying to play with GPT model and prepare it for deployment to production using the ONNX runtime.

<Speak bout the onnx runtime>, but in I failed to make it work on ONNX runtime with the beam search capacity.
I looked at the internet I couldn't find a tutorial where there is an explanation on how to implement beam search on an onnx, but I was able to find out that the onnx runtime has a beam search nodes that designed for that purpose.

I decided to give it a try with a custom model and in this post I will describe how it wen.

### The model.

In my experiment, I was building a Retrieval Augmented generation model for medical science and I was using the BioGPT model.
Model checkpoint were not released in Huggingface, I had to play with the fairseq model checkpoint and save them to huggingface.

After that I started with the model conversion to onnx. It was a bit of a hassle because the conversion script in the onnx runtime was designed for the main GPT2 model but the biogpt model was slightly different from the main GPT2 model, I had to adjust the code in the for the main gpt2 model to make it work for my use case. 

Let us see how it went:

### Importing the model

I had the model saved locally on my laptop, so my first step was to import it

Below is the code that I use for the import:

```python
from pathlib import Path

model_path = Path.cwd().joinpath('models')
model_id = 'bio-gpt-qa'
model_path = model_path.joinpath(model_id)
```

And then I imported the model with 

```python

from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed


tokenizer = BioGptTokenizer.from_pretrained(model_path,  local_files_only=True)
model = BioGptForCausalLM.from_pretrained(model_path, local_files_only=True)
```


The tokenization was done with the following lines:

```python
input = f"'question:what is the cause of covid ? context: the cause of covid is a virus'"
encoded_input = tokenizer([input],
                          return_tensors='pt',
                          max_length=1024,
                          truncation=True)
```


With that we could generate the output given the input with:

```python
generate_tokens = model.generate(**encoded_input,
                                 num_beams=5,
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95,
                                 max_length=512)
```


And decode the token back and print them:

```python
generated_text = tokenizer.decode(generate_tokens[0], skip_special_tokens=True)
print(generated_text)
```


### Exporting the Model to ONNX

In this section we will use the optimun library to export the model to ONNX format. We will not only export the model graph with the standard inputs and outputs but we will also be adding the past decoder state which will help to speed up the generation with beam search.
<ADD a link about the beam search here>

There are two ways to export the model via onnx:

The optimum library have an easier way to export a model, you can export the model with a single command line argument and have the exported model. However for my use case, this approach fails because It couldn't pass the validation defined in the onnx runtime. As of now I am still finding how to tweak it to pass those validation.

The onnxruntime approach, this one was a little bit complicate to customize, I had to go inside the library code an change some bits here and there, to make it work. If this work the model could pass the validation check for the ONNX runtime.

I had to use both approach , the optimum library to export the model with the past and the onnxruntime approach to add the beam search node to my model.


#### Exporting the model with past

To export the model to onnx we need to first define the  model inputs and outputs shapes and create some dummy input the model will use to create a computational graph to generate the onnx model.

In our case those input and output were defined in the following config:

```python
from optimum.exporters.onnx.model_configs import GPT2OnnxConfig
from typing import Dict, OrderedDict, Any, List




from transformers import PretrainedConfig


class CustomBioGPTConfig(GPT2OnnxConfig):

    def __init__(self, config: PretrainedConfig, 
                 task: str = "text-generation-with-past", 
                 int_dtype: str = "int32", 
                 float_dtype: str = "fp16", 
                 use_past: bool = True, 
                 use_past_in_inputs: bool = True, 
                 preprocessors: List[Any] | None = None, legacy: bool = False):
        super().__init__(config, task, int_dtype, float_dtype, use_past, use_past_in_inputs, preprocessors, legacy)
        print("the int dtype is ", int_dtype)
        self._config.n_layer = config.num_hidden_layers
        self._config.n_head = config.num_attention_heads
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {"input_ids": {0: "batch_size", 1: "sequence"}, 
                        "attention_mask": {0: "batch_size", 1: "sequence_length"}}

        
        self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs
    
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence"}})
        self.add_past_key_values(common_outputs, direction="outputs")
    
        return common_outputs
    
    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """
        Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

        Args:
            inputs_or_outputs (`Dict[str, Dict[int, str]]`):
                The mapping to fill.
            direction (`str`):
                either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                output mapping, this is important for axes naming.
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(
                f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_seq_len"
            name = "past_key_values"
        else:
            decoder_sequence_name = "total_seq_len"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {
                0: "batch_size", 3: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {
                0: "batch_size", 3: decoder_sequence_name}
```

The above class setup the onnx inputs and outputs for the models, we can see that we set up the input names, the shape. The model inputs are:

- input_ids, the attention_masks, and the past key values.

The outputs are the logits and the present key values.

We also specify the dynamic axes, or the axes in the inputs or output shapes that changes according to the query. We can see that for our element, the batch size and the sequence length are dynamics. They are expected to change with the query.

Note that my main model was not taking the position_ids but the validation check in the onnxruntime requires it. So I have to tweak my model class to use it.


Let us now use that to generate the inputs:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path, local_files_only=True)
custom_config = CustomBioGPTConfig(config=config)

```


And then the final export script

```python
from optimum.exporters.onnx import main_export

main_export(
model_name_or_path=model_path,
task="text-generation-with-past",
model_kwargs={"output_attentions": True},
output=onnx_path.joinpath('bio-gpt-model-with-past'),
custom_onnx_configs={"model": custom_config},
)
```


If we look at the main export function, it takes the model path, the task , the model kwargs and the custom config.

All those parameters are self explanatory, but for our case we need to adjust the model kwargs and set it to return the output attention to make it work for the case.

This code generate the model ONNX with inputs and output shapes. that model was saved at the specified path and we could use it. 

However to make it work with the ONNX runtime we need to add the beam search node to it.

For that purpose we will use the the onnxruntime tools to export the model with beam search..


### Adding the beamsearch node to the model.

Tommorow.


```python
args_list = ['--model_name_or_path',
 '/Users/esp.py/Projects/Personal/end-to-end-rag/models/bio-gpt-qa',
 '--output',
 '/Users/esp.py/Projects/Personal/end-to-end-rag/models_repository/generator/generator_model/1/biogpt-model-with-past-and-beam.onnx',
 '--model_type',
 'gpt2',
 '--num_beams',
 '5',
 '--temperature',
 '0.25',
 '--model_class',
 'BioGptModel']
```
This code will take the model and export
I will generate the model with past and then export it to ONNX.

```args = parse_arguments(arguments_list)

from onnxruntime.transformers.convert_generation  import convert_generation_model, parse_arguments, GenerationType
```
And then this code will convert the model

```python
convert_generation_model(args=args, generation_type=GenerationType.BEAMSEARCH)
```

Remember to add dhat to export the model we had to adjust the code of the gpt_helper and add the following: 


```python
class MyBioGptModel(BioGptForCausalLM):
    def forward(self, input_ids, position_ids, attention_mask, *past):
        attention_mask = attention_mask + position_ids - position_ids # just to make sure we are using the position id in the garph.
        results = super().forward(input_ids, 
                               attention_mask=attention_mask, 
                               past_key_values=past, 
                               output_attentions=True)
        return MyBioGptModel.post_process(results, self.config.num_hidden_layers)

    @staticmethod
    def post_process(result, num_layer):
        if isinstance(result[1][0], (tuple, list)):
            assert len(result[1]) == num_layer and len(result[1][0]) == 2
            # assert len(result[1][0][0].shape) == 4 and result[1][0][0].shape == result[1][0][1].shape
            present = []
            for i in range(num_layer):
                # Since transformers v4.*, past key and values are separated outputs.
                # Here we concate them into one tensor to be compatible with Attention operator.
                present.append(
                    torch.cat(
                        (result[1][i][0].unsqueeze(0), result[1][i][1].unsqueeze(0)),
                        dim=0,
                    )
                )
            return (result[0], tuple(present))

        return result
```

And in the model class we added: 

`  "BioGptModel": (MyBioGptModel, "logits", True)`

# Note the decoder onnx path

This code will export the model to onnx runtime with the beam search nodes, but it fails to pass the validation checks. NEed to come back and test
