export TOOLBENCH_KEY="Set your toolbench key here"
export OPENAI_KEY="Set your openai api key here"
export PYTHONPATH=./
export SERVICE_URL="http://localhost:8080/virtual"
export OUTPUT_DIR="data/answer/llama-3-toolllama-g123-cot-opendomain"
export CUDA_VISIBLE_DEVICES=0
model_path="../models/ToolLlama/llama-3-toolllama-g123"
stage="G3"
group="instruction"


# Open domain setting
corpus_tsv_path="data/retrieval/${stage}/corpus.tsv"
retrieval_model_path="../models/ToolLlama/retriever/bert_${stage}"


mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

# Open Domain
cmd="python toolbench/inference/qa_pipeline_multithread.py \
    --model_path ${model_path} \
    --tool_root_dir data/toolenv/tools \
    --corpus_tsv_path ${corpus_tsv_path} \
    --retrieval_model_path ${retrieval_model_path} \
    --backbone_model toolchat \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 \
    --function_provider retriever --overwrite"

echo $cmd
eval $cmd