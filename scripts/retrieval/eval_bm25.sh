# model="bert_G1_retrieval"
# ../models/ToolLlama/retriever/${model}
# model="bm25"
# stage="G1"
# split="tool"
# result_path="bm25"

# cmd="python -m evaluation.eval_bm25 \
#     --model_name_or_path ${model} \
#     --stage ${stage} \
#     --split ${split} \
#     --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\""

# echo $cmd
# eval $cmd

model="bm25"
stage="G3"
split="instruction"
result_path="bm25"
corpus="G123"

cmd="python -m evaluation.eval_bm25 \
    --model_name_or_path ${model} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" \
    --corpus ${corpus}"

echo $cmd
eval $cmd