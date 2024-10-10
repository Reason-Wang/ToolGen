![banner.png](assets/banner.png)
# ToolGen: Unified Tool Retrieval and Calling via Generation

<p align="center">
    <a href="https://huggingface.co/collections/reasonwang/toolgen-668a46a4959745ec8e9891f6">🤗ToolGen Model </a>
    • 
	<a href="https://arxiv.org/pdf/2410.03439">📄Paper (arxiv)</a>
	• 
    <a href="https://huggingface.co/datasets/reasonwang/ToolGen-Datasets">🤗ToolGen Datasets</a>
</p>

Integrating tool knowledge directly into LLMs by representing tools as unique tokens, enabling seamless tool invocation and language generation. 🔧🤖

<!--  Currently busy now, will update the repo as soon as possible... -->

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

More details will be updated soon.

## Citation
If this work is helpful, please kindly cite as:
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

<!-- ## Training
### Tool Memorization

### Retrieval Training

### End-to-End Agent-Tuning



## Evaluation

### Retrieval

### Inference

### Solvable Pass Rate

### Solvable Win Rate -->