model="bm25"
stage="G1"
split="instruction" # instruction, tool, category
result_path="bm25"
corpus="G123" # G123, G1, G2, G3. G123 is the multi-domain setting

cmd="python -m evaluation.retrieval.eval_bm25 \
    --model_name_or_path ${model} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" \
    --corpus ${corpus}"

echo $cmd
eval $cmd