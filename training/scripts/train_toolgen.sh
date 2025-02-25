# Train tool memorization
pretrain_dir="meta-llama/Meta-Llama-3-8B"
checkpoint_dir="checkpoints/ToolGen-Llama-3-8B-Tool-Memorization"
flash_attention="True"
run_name="llama-3-8b-tool-memorization"
datasets="toolgen_atomic_memorization.json"
dataset_nums="10000000"
max_length="1024"
batch_size="2"
lr="2e-5"
accumulation_steps="64"
epochs="8"
add_virtual_tokens="True"
template="llama-3"
save_strategy="steps"
save_steps="1000"
zero="z2"

# Train tool retrieval
# pretrain_dir="checkpoints/ToolGen-Llama-3-8B-Tool-Memorization"
# checkpoint_dir="checkpoints/ToolGen-Llama-3-8B-Tool-Retriever"
# flash_attention="True"
# run_name="llama-3-8b-tool-retrieval"
# datasets="toolgen_atomic_retrieval_G123.json"
# dataset_nums="1000000"
# max_length="1024"
# batch_size="2"
# lr="2e-5"
# accumulation_steps="64"
# epochs="1"
# add_virtual_tokens="False"
# template="llama-3"
# save_strategy="steps"
# save_steps="1000"
# zero="z2"

# End2End
# pretrain_dir="checkpoints/ToolGen-Llama-3-8B-Tool-Retriever"
# checkpoint_dir="checkpoints/ToolGen-Llama-3-8B"
# flash_attention="True"
# run_name="llama-3-8b-end2end"
# datasets="toolgen_atomic_G123_dfs.json"
# dataset_nums="10000000"
# max_length="6144"
# batch_size="1"
# lr="2e-5"
# accumulation_steps="64"
# epochs="1"
# add_virtual_tokens="False"
# template="llama-3"
# save_strategy="steps"
# save_steps="1000"
# zero="z3_offload"

chat="True"

cmd="deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25024 train.py \
  --model_name_or_path ${pretrain_dir} \
  --add_virtual_tokens ${add_virtual_tokens} \
  --flash_attention ${flash_attention} \
  --deepspeed src/configs/ds_${zero}_config.json \
  --chat ${chat} \
  --template ${template} \
  --architecture causal \
  --output_dir ${checkpoint_dir} \
  --save_strategy ${save_strategy} \
  --save_steps ${save_steps} \
  --gather_weights True \
  --learning_rate ${lr} \
  --warmup_ratio 0.03 \
  --datasets ${datasets} \
  --dataset_nums ${dataset_nums} \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${accumulation_steps} \
  --max_length ${max_length} \
  --num_train_epochs ${epochs} \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name ${run_name}"

echo $cmd
eval $cmd