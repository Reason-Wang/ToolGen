export OPENAI_API_KEY=""
model="gpt-4o"
stage="G3" # G1, G2, G3
split="test"
corpus="G3" # G123, G1, G2, G3. G123 is the multi-domain setting
result_path="GPT-4o"
cmd="python -m evaluation.retrieval.eval_longcontext \
    --model_name_or_path ${model} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" 
    --corpus ${corpus}"

echo $cmd
eval $cmd