export TOOLBENCH_KEY="Set your toolbench key here"
export OPENAI_KEY="Set you openai api key here"
export PYTHONPATH=./
chatgpt_model="gpt-4o"
export SERVICE_URL="http://localhost:8080/virtual"
export OUTPUT_DIR="data/answer/test"
export CUDA_VISIBLE_DEVICES=0

model_path="reasonwang/ToolLlama-Llama-3-8B"
stage="G2"
group="instruction"


# Open domain setting
corpus_tsv_path="data/retrieval/${stage}/corpus.tsv"
# retrieval_model_path="../models/ToolLlama/retriever/bert_${stage}"
retrieval_model_path="ToolBench/ToolBench_IR_bert_based_uncased"


mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

# Open Domain
cmd="python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --model_path ${model_path} \
    --tool_root_dir data/toolenv/tools \
    --chatgpt_model ${chatgpt_model} \
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
    --function_provider retriever"

echo $cmd
eval $cmd