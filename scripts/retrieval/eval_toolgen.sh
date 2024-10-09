# model_name_or_path="../models/VAgent/llama-3-8b-tool-semantic-epoch8-retrieval"
# indexing="Semantic"
# constrain="False"
export CUDA_VISIBLE_DEVICES=3
model_name_or_path=""reasonwang/VAgent-Llama-3-8B-Tool-Retrieval-Numeric""
indexing="Numeric"
constrain="False"

# model_name_or_path="../models/VAgent/llama-3-8b-tool-hierarchical-epoch8-retrieval"
# indexing="Hierarchical"
# constrain="False"

# model_name_or_path="reasonwang/VAgent-Llama-3-8B-Tool-Retrieval"
# indexing="Atomic"
# constrain="False"

# model_name_or_path="../models/VAgent/llama-3-8b-tool-retrieval-womemorization"
# indexing="Atomic"
# constrain="False"

# model_name_or_path="reasonwang/VAgent-Llama-3-8B-Tool-Memorization"
# indexing="Atomic"
# constrain="True"

stage="G3"
split="instruction"

cmd="python -m evaluation.eval_toolgen \
    --model_name_or_path ${model_name_or_path} \
    --indexing ${indexing} \
    --stage ${stage} \
    --split ${split} \
    --result_path data/results/retrieval/ \
    --constrain ${constrain}"
echo $cmd
eval $cmd