export TOOLBENCH_KEY="Set your toolbench key here"
export OPENAI_KEY="Set your openai api key here"
export PYTHONPATH=./
export SERVICE_URL="http://localhost:8080/virtual"


export GPT_MODEL="gpt-3.5-turbo-16k"
export OUTPUT_DIR="data/answer/test"
group="G2_instruction"

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir data/toolenv/tools \
    --backbone_model chatgpt_function \
    --chatgpt_model $GPT_MODEL \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file data/solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1  --function_provider "truth"