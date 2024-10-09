# model_name_or_path="text_embedding_large"
# stage="G1"
# split="instruction"
# result_path="openai"
# api_key="Set your openai api key here"

# cmd="python -m evaluation.eval_openai_embedding \
#     --model_name_or_path ${model_name_or_path} \
#     --api_key ${api_key} \
#     --stage ${stage} \
#     --split ${split} \
#     --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\""

# echo $cmd
# eval $cmd

model_name_or_path="text_embedding_large"
stage="G3"
split="instruction"
result_path="openai"
api_key="Set your openai api key here"
corpus="G123"

cmd="python -m evaluation.eval_openai_embedding \
    --model_name_or_path ${model_name_or_path} \
    --api_key ${api_key} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" \
    --corpus ${corpus}"

echo $cmd
eval $cmd