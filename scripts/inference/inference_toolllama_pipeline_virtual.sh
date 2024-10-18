export TOOLBENCH_KEY="Set your toolbench key here"
export OPENAI_KEY="Set your openai api key here"
export PYTHONPATH=./
export SERVICE_URL="http://localhost:8080/virtual"
chatgpt_model="gpt-4o"
export OUTPUT_DIR="data/answer/test"

model_path="reasonwang/ToolLlama-Llama-3-8B"
stage="G2"
group="instruction"


mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
cmd="python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --model_path ${model_path} \
    --tool_root_dir data/toolenv/tools \
    --chatgpt_model ${chatgpt_model} \
    --backbone_model toolchat \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 --function_provider truth"

echo $cmd
eval $cmd