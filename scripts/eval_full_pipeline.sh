export TOOLBENCH_KEY="Put your ToolBench key here"
export OPENAI_KEY="Put your OpenAI key here"
export PYTHONPATH=./
export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"

# MODEL_NAME=virtual-gpt35-16k-step16-cot
# model_path="virtual-gpt35-16k-step16-cot"
# backbone_model="chatgpt_function"
# function_provider="truth"

# MODEL_NAME="ToolLlama-v2-t0.0-cot"
# model_path="ToolBench/ToolLLaMA-2-7b-v2"
# indexing="None"
# function_provider="truth"
# backbone_model="toolllama"

# MODEL_NAME="ToolLlama-Llama-3-8B-cot"
# model_path="reasonwang/ToolLlama-Llama-3-8B"
# function_provider="truth"
# backbone_model="toolchat"


# MODEL_NAME="ToolGen-Semantic-Llama-3-8B-cot"
# model_path="reasonwang/ToolGen-Semantic-Llama-3-8B"
# indexing="Semantic"


# model_path="reasonwang/ToolGen-Llama-3-8B-Instruct"
# indexing="Atomic"
# template="llama-3"
# MODEL_NAME="ToolGen-Llama-3-8B-Instruct"


# model_code="Qwen2.5-14B"
# model_path="reasonwang/ToolGen-${model_code}"
# indexing="Atomic"
# template="qwen-7b-chat"
# MODEL_NAME="ToolGen-${model_code}-WoRetry"
# function_provider="all"

if [ $indexing == "Atomic" ]; then
    backbone_model="toolgen_atomic"
else
    list=("Semantic" "Numeric" "Hierarchical")
    for item in "${list[@]}"; do
        if [ "$item" = "$indexing" ]; then
            backbone_model="toolgen"
            break
        fi
    done
fi


export CUDA_VISIBLE_DEVICES=4
OUTPUT_DIR="data/answer/${MODEL_NAME}"
stage="G2"
group="instruction"
method="CoT@1"

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/${stage}_${group}
cmd_inference="python evaluation/toolbench/inference/qa_pipeline_multithread.py \
    --model_path ${model_path} \
    --template ${template} \
    --indexing ${indexing} \
    --chatgpt_model ${GPT_MODEL} \
    --tool_root_dir data/toolenv/tools \
    --backbone_model ${backbone_model} \
    --openai_key ${OPENAI_KEY} \
    --max_observation_length 1024 \
    --method ${method} \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key ${TOOLBENCH_KEY} \
    --num_thread 1  \
    --function_provider ${function_provider}"

echo $cmd_inference
eval $cmd_inference


RAW_ANSWER_PATH="data/answer"
CONVERTED_ANSWER_PATH="data/model_predictions_converted"

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir="${RAW_ANSWER_PATH}/${MODEL_NAME}/${stage}_${group}"
output_file="${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${stage}_${group}.json"
echo ${output_file}
cmd_convert="python -m evaluation.toolbench.tooleval.convert_to_answer_format\
    --answer_dir ${answer_dir} \
    --method ${method} \
    --output ${output_file}"

echo $cmd_convert
eval $cmd_convert

export API_POOL_FILE=openai_keys.json
SAVE_PATH="data/results/pass_rate"
mkdir -p ${SAVE_PATH}
export EVAL_MODEL=gpt-4-turbo-2024-04-09
mkdir -p ${SAVE_PATH}/${MODEL_NAME}

cmd_pass="python -m evaluation.toolbench.tooleval.eval_pass_rate \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --reference_model ${MODEL_NAME} \
    --test_ids data/solvable_queries/test_query_ids \
    --max_eval_threads 3 \
    --evaluate_times 3 \
    --test_set ${stage}_${group}"

echo $cmd_pass
eval $cmd_pass

export API_POOL_FILE=openai_keys.json
SAVE_PATH="data/results/preference_rate"
PASS_RATE_PATH="data/results/pass_rate"
REFERENCE_MODEL=virtual-gpt35-16k-step16-cot
export EVAL_MODEL=gpt-4o-2024-05-13
mkdir -p ${SAVE_PATH}

cmd_preference="python -m evaluation.toolbench.tooleval.eval_preference \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${MODEL_NAME} \
    --test_ids data/solvable_queries/test_query_ids/ \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --pass_rate_result_path ${PASS_RATE_PATH} \
    --max_eval_threads 3 \
    --use_pass_rate true \
    --evaluate_times 3 \
    --test_set ${stage}_${group}"

echo $cmd_preference
eval $cmd_preference