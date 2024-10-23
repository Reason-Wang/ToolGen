![banner.png](assets/banner.png)
# ToolGen: Unified Tool Retrieval and Calling via Generation

<p align="center">
    <a href="https://huggingface.co/collections/reasonwang/toolgen-668a46a4959745ec8e9891f6">🤗ToolGen Model </a>
    • 
	<a href="https://arxiv.org/pdf/2410.03439">📄Paper (arxiv)</a>
	<!-- •  -->
    <!-- <a href="https://huggingface.co/datasets/reasonwang/ToolGen-Datasets">🤗ToolGen Datasets</a> -->
</p>

ToolGen is a framework that integrates tool knowledge directly into LLMs by representing tools as unique tokens, enabling seamless tool invocation and language generation.🔧🦙 With 47,000 tool tokens, ToolGen shows superior performance in both tool retrieval and task completion.


## Run ToolGen

The following code snippet shows how to run ToolGen locally. First, get your ToolBench key from [ToolBench](https://github.com/OpenBMB/ToolBench) repo. Then deploy [StableToolBench](https://github.com/THUNLP-MT/StableToolBench) following the instructions in their repo.

```python
import json
from OpenAgent.agents.toolgen.toolgen import ToolGen
from OpenAgent.tools.src.rapidapi.rapidapi import RapidAPIWrapper

# Initialize rapid api tools
with open("keys.json", 'r') as f:
    keys = json.load(f)
toolbench_key = keys['TOOLBENCH_KEY']
rapidapi_wrapper = RapidAPIWrapper(
    toolbench_key=toolbench_key,
    rapidapi_key="",
)

toolgen = ToolGen(
    "reasonwang/ToolGen-Llama-3-8B",
    indexing="Atomic",
    tools=rapidapi_wrapper,
)

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "I'm a football fan and I'm curious about the different team names used in different leagues and countries. Can you provide me with an extensive list of football team names and their short names? It would be great if I could access more than 7000 team names. Additionally, I would like to see the first 25 team names and their short names using the basic plan."}
]

toolgen.restart()
toolgen.start(
    single_chain_max_step=16,
    start_messages=messages
)

```
## ToolGen
Download and decompress [data.tar.gz](https://huggingface.co/datasets/reasonwang/ToolGen-Datasets/blob/main/data.tar.gz). Other datasets are at [🤗ToolGen-Datasets](https://huggingface.co/datasets/reasonwang/ToolGen-Datasets).

### Tool Virtualization
The first step is to map tools into tokens. We have extracted all the tools in ToolBench and converted them into tokens, as shown in [virtual_tokens.txt](data/virtual_tokens.txt). The following code adds the tokens into the vocabulary and expands model embeddings.

```python
with open('data/virtual_tokens.txt', 'r') as f:
    virtual_tokens = f.readlines()
    virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]

model_name_or_path = "meta-llama/Meta-Llama-3-8B"
# Load tokenizer and add tokens into vocabulary
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
# Load model and expand embeddings
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))
combined_tokens = []
for vt in virtual_tokens:
    combined_token = vt[2:-2].split("&&")
    combined_tokens.append(combined_token)
    
for combined_token, virtual_token in zip(combined_tokens, virtual_tokens):
    combined_token_ids = tokenizer(" ".join(combined_token), add_special_tokens=False).input_ids
    virtual_token_id = tokenizer(virtual_token, add_special_tokens=False).input_ids
    assert len(virtual_token_id) == 1
    combined_token_embeddings = model.model.embed_tokens(torch.tensor(combined_token_ids).to(model.device))
    embedding = torch.mean(combined_token_embeddings, dim=0)
    model.model.embed_tokens.weight.data[virtual_token_id[0]] = embedding
```

### Tool Memorization
After tool virtualization, there is a three-stage training to finetune ToolGen. The first stage is tool memorization, which trains the model to memorize all tool tokens. The data for this stage is at [🤗ToolGen-Memorization](https://huggingface.co/datasets/reasonwang/ToolGen-Datasets/blob/main/toolgen_atomic_memorization.json). We have converted the format into ShareGPT-like format for an easy integration with current training framework like [FastChat](https://github.com/lm-sys/FastChat) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Note that we train the first stage for 8 epochs. A sample is shown bellow:
```
{
    "conversations": [
        {
            "role": "user",
            "content": "Tool Name: QRCheck. Tool Description: Check the quality of any QRCode Api Name: quality_v1_quality_post Api Description: None.",
            "loss": false
        },
        {
            "role": "assistant",
            "content": "<<QRCheck&&quality_v1_quality_post>>",
            "loss": true
        }
    ]
}
```

### Retrieval Training
The second stage mainly trains the tool retrieval capability of ToolGen. The data is also at [🤗ToolGen-Retrieval](https://huggingface.co/datasets/reasonwang/ToolGen-Datasets/blob/main/toolgen_atomic_retrieval_G123.json). We train it for 1 epoch. After the second stage training, we obtain ToolGen-Retriever. A sample is shown below:
```
{
    "conversations": [
        {
            "role": "user",
            "content": "My friends and I are organizing a hackathon on 'web development' and 'mobile app development'. We need some inspiration and guidance. Can you fetch the top stories on these topics from Medium.com?",
            "loss": false,
        },
        {
            "role": "assistant",
            "content": "<<Medium&&/search/topics>>",
            "loss": true
        }
    ]
}
```
### End-to-End Agent-Tuning
Finally, we train the ToolGen with agent trajectories to enable them task completion capability. The data is at [🤗ToolGen-Agent](https://huggingface.co/datasets/reasonwang/ToolGen-Datasets/blob/main/toolgen_atomic_G123_dfs.json).


## Evaluation
### Retrieval
The following command shows an example to evaluate the retrieval performance. Other tool retrieval evaluation scripts can be found in `scripts/retrieval`.

```
python -m evaluation.retrieval.eval_toolgen \
    --model_name_or_path "reasonwang/ToolGen-Llama-3-8B-Tool-Retriever" \
    --indexing "Atomic" \
    --stage "G1" \
    --split "instruction" \
    --result_path data/results/retrieval/ \
    --constrain True
```

### Inference
For end-to-end evaluation, first get [ToolBench](https://github.com/OpenBMB/ToolBench) Key and run [StableToolBench](https://github.com/THUNLP-MT/StableToolBench).
Then, perform inference on queries to generate trajectories. Scripts can be found in `scripts/inference`

### Solvable Pass Rate
First, using `scripts/convert_answer/run_convert_answer.sh` to convert trajectory format. Then run `scripts/pass_rate/run_pass_rate.sh` for pass rate evaluation.

### Solvable Win Rate
Run `scripts/preference/run_preference.sh` for win rate evaluation.

## Citation
If our work is helpful, please kindly cite as:
```
@misc{wang2024toolgenunifiedtoolretrieval,
      title={ToolGen: Unified Tool Retrieval and Calling via Generation}, 
      author={Renxi Wang and Xudong Han and Lei Ji and Shu Wang and Timothy Baldwin and Haonan Li},
      year={2024},
      eprint={2410.03439},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03439}, 
}
```

