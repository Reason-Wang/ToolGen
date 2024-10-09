# model="bert_G1_retrieval"
# ../models/ToolLlama/retriever/${model}
# model="ToolBench/ToolBench_IR_bert_based_uncased"
# # model="reasonwang/BERT-G3"
# stage="G1"
# split="test"
# result_path="bert_G3_retrieval"
# # replace_file="../datasets/toolgen/G3_instruction_new_queries.json"

# cmd="python -m evaluation.eval_encoder_efficiency \
#     --model_name_or_path ${model} \
#     --stage ${stage} \
#     --split ${split} \
#     --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\""

# echo $cmd
# eval $cmd

# model="ToolBench/ToolBench_IR_bert_based_uncased"
export CUDA_VISIBLE_DEVICES=4
model="reasonwang/BERT-G3"
stage="G2"
split="category"
corpus="G123"
result_path="BERT-G1-full-tools"
cmd="python -m evaluation.eval_encoder \
    --model_name_or_path ${model} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" 
    --corpus ${corpus}"

echo $cmd
eval $cmd