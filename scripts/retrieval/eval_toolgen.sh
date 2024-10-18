model_name_or_path="reasonwang/ToolGen-Llama-3-8B-Tool-Retriever"
indexing="Atomic"
constrain="True"

stage="G1" # G1, G2, G3
split="instruction" # instruction, tool, category

cmd="python -m evaluation.retrieval.eval_toolgen \
    --model_name_or_path ${model_name_or_path} \
    --indexing ${indexing} \
    --stage ${stage} \
    --split ${split} \
    --result_path data/results/retrieval/ \
    --constrain ${constrain}"
echo $cmd
eval $cmd