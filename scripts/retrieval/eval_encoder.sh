# model="reasonwang/BERT-G3"
model="ToolBench/ToolBench_IR_bert_based_uncased"
stage="G1" # G1, G2, G3
split="instruction"
corpus="G123" # G123, G1, G2, G3. G123 is the multi-domain setting
result_path="BERT-G1-full-tools"
cmd="python -m evaluation.retrieval.eval_encoder \
    --model_name_or_path ${model} \
    --stage ${stage} \
    --split ${split} \
    --result_path \"data/results/retrieval/${result_path}_${stage}_${split}.json\" 
    --corpus ${corpus}"

echo $cmd
eval $cmd