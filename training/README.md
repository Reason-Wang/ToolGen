## Training

Training requires DeepSpeed as dependency:
```
pip install deepspeed
```


### Tool Memorization
In the first stage, we use the following command to train ToolGen. The LLM (Llama-3-8B in this case) is first added tool tokens then expanded embeddings, which is controled by `add_virtual_tokens` argument. 

```bash
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --add_virtual_tokens True \
  --flash_attention True \
  --deepspeed src/configs/ds_z2_config.json \
  --chat True \
  --template llama-3 \
  --architecture causal \
  --output_dir checkpoints/ToolGen-Llama-3-8B-Tool-Memorization \
  --save_strategy steps \
  --save_steps 1000 \
  --gather_weights True \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --datasets toolgen_atomic_memorization.json \
  --dataset_nums 10000000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --max_length 1024 \
  --num_train_epochs 8 \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name llama-3-8b-tool-memorization
```

### Tool Retrieval
In the second stage, we train the ToolGen model with queries and tool tokens, intialized from the model obtained in the first stage. Since the model is already added tool tokens and expanded embeddings, we set `add_virtual_tokens` to `False`.
```bash
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path checkpoints/ToolGen-Llama-3-8B-Tool-Memorization \
  --add_virtual_tokens False \
  --flash_attention True \
  --deepspeed src/configs/ds_z2_config.json \
  --chat True \
  --template llama-3 \
  --architecture causal \
  --output_dir checkpoints/ToolGen-Llama-3-8B-Tool-Retriever \
  --save_strategy steps \
  --save_steps 1000 \
  --gather_weights True \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --datasets toolgen_atomic_retrieval_G123.json \
  --dataset_nums 1000000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --max_length 1024 \
  --num_train_epochs 1 \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name llama-3-8b-tool-retrieval
```

### End-to-End Training
In the last stage, we train the ToolGen agent model with end-to-end trajectories. We set the maximum length to 6144, which generally needs large GPU memory. Based on our experiments, 4 GPUs each with 80GB memory are enough for this stage (Deepspeed zero 3 with offloading is used).
```bash
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path checkpoints/ToolGen-Llama-3-8B-Tool-Retriever \
  --add_virtual_tokens False \
  --flash_attention True \
  --deepspeed src/configs/ds_z3_offload_config.json \
  --chat True \
  --template llama-3 \
  --architecture causal \
  --output_dir checkpoints/ToolGen-Llama-3-8B \
  --save_strategy steps \
  --save_steps 1000 \
  --gather_weights True \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --datasets toolgen_atomic_G123_dfs.json \
  --dataset_nums 10000000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --max_length 6144 \
  --num_train_epochs 1 \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name llama-3-8b-end2end
```
