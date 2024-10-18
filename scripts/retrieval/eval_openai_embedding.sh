model_name_or_path="text_embedding_large"
stage="G1" # G1, G2, G3
split="instruction"
result_path="openai"
api_key="Set your openai api key here"
corpus="G123" # G123, G1, G2, G3. G123 is the multi-domain setting

cmd="python -m evaluation.retrieval.eval_openai_embedding \
    --model_name_or_path ${model_name_or_path} \
    --api_key ${api_key} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" \
    --corpus ${corpus}"

echo $cmd
eval $cmd