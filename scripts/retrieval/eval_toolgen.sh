model_name_or_path="reasonwang/ToolGen-Llama-3-8B-Tool-Retriever"
indexing="Atomic"
constrain="True"
limit_to_stage_space="False"
template="llama-3"

stage="G1" # G1, G2, G3
split="instruction" # instruction, tool, category

cmd="python -m evaluation.retrieval.eval_toolgen \
    --model_name_or_path ${model_name_or_path} \
    --indexing ${indexing} \
    --stage ${stage} \
    --split ${split} \
    --result_path data/results/retrieval/ \
    --constrain ${constrain} \
    --limit_to_stage_space ${limit_to_stage_space} \
    --template ${template}"
echo $cmd
eval $cmd