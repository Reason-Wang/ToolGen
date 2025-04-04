export TOOLBENCH_KEY="Set your ToolBench key here"
export OPENAI_KEY="Set your OpenAI key here"
export PYTHONPATH=./
export SERVICE_URL="http://localhost:8080/virtual"
export CUDA_VISIBLE_DEVICES=0

model_path="reasonwang/ToolGen-Llama-3-8B"
indexing="Atomic"
template="llama-3"

# model_name="Qwen2.5-3B"
# model_path="reasonwang/ToolGen-${model_name}"
# indexing="Atomic"
# template="qwen-7b-chat"

export OUTPUT_DIR="data/answer/${model_name}/"
stage="G3"
group="instruction"

if [ $indexing == "Atomic" ]; then
    backbone_model="toolgen_atomic"
else
    backbone_model="toolgen"
fi

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/${stage}_${group}
python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --chatgpt_model gpt-4o \
    --model_path ${model_path} \
    --template ${template} \
    --indexing ${indexing} \
    --tool_root_dir data/toolenv/tools \
    --backbone_model ${backbone_model} \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 \
    --function_provider all