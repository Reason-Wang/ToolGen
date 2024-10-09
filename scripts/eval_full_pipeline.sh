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

# MODEL_NAME="ToolLlama-Llama-3-8B-t0.0-cot"
# model_path="reasonwang/ToolLlama-Llama-3-8B"
# function_provider="truth"
# backbone_model="toolchat"


# MODEL_NAME="ToolGen-Atomic-WTokens-Llama-3-8B-t0.0-cot"
# model_path="../models/VAgent/toolgen-llama-3-8b-tool-wplanning-wtokens-half"

# MODEL_NAME="ToolGen-Atomic-Planning-Prefix-Llama-3-8B-t0.0-cot-retry-finish"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="ToolGen-Atomic-Epoch4-Mixed-Llama-3-8B-t0.0-cot"
# model_path="../models/VAgent/llama-3-8b-tool-mixed"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-t0.0-cot-replace"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"
# replace_file="../datasets/toolgen/G3_instruction_new_queries.json"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-t0.0-cot-noconstrain-retry-finish"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-t0.0-cot-retry-finish"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="test"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-t0.0-cot-gpt4o-answer-rewrite"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="ToolGen-Atomic-Filter-Llama-3-8B-t0.0-cot"
# model_path="../models/VAgent/ToolGen-Atomic-Llama-3-8B-Tool-Planning-Filter"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-t0.0-cot-unconstrain"
# model_path="reasonwang/VAgent-Llama-3-8B-Planning-Full"

# MODEL_NAME="ToolGen-Atomic-Llama-3-8B-Tool-WoPlanning-t0.0-cot"
# model_path="../models/VAgent/ToolGen-Atomic-Llama-3-8B-Tool-WoPlanning"

# indexing="Atomic"
# function_provider="all"
# backbone_model="vagent"

# MODEL_NAME="ToolGen-Semantic-Planning-Llama-3-8B--t0.0-cot"
# model_path="reasonwang/ToolGen-Semantic-Llama-3-8B-Tool-Planning"

# MODEL_NAME="ToolGen-Numeric-Planning-Llama-3-8B--t0.0-cot"
# model_path="../models/VAgent/ToolGen-Numeric-Llama-3-8B-Tool-Planning"

# MODEL_NAME="ToolGen-Hierarchical-Planning-Llama-3-8B--t0.0-cot"
# model_path="../models/VAgent/ToolGen-Hierarchical-Llama-3-8B-Tool-Planning"

# MODEL_NAME="ToolGen-Semantic-Planning-Llama-3-8B--t0.0-cot-unconstrain-retry-finish"
# model_path="reasonwang/ToolGen-Semantic-Llama-3-8B-Tool-Planning"
# indexing="Semantic"

# MODEL_NAME="ToolGen-Semantic-Planning-Llama-3-8B--t0.0-cot-unconstrain"
# model_path="reasonwang/ToolGen-Semantic-Llama-3-8B-Tool-Planning"
# indexing="Semantic"

MODEL_NAME="ToolGen-Numeric-Planning-Llama-3-8B--t0.0-cot-unconstrain-retry-finish"
indexing="Numeric"
model_path="../models/VAgent/ToolGen-Numeric-Llama-3-8B-Tool-Planning"
function_provider="all"
backbone_model="toolgen"

export CUDA_VISIBLE_DEVICES=2
OUTPUT_DIR="data/answer/${MODEL_NAME}"
stage="G1"
group="instruction"
method="CoT@1"

mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/${stage}_${group}
cmd="python toolbench/inference/qa_pipeline_multithread.py \
    --model_path ${model_path} \
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
echo $cmd
eval $cmd


RAW_ANSWER_PATH="data/answer"
CONVERTED_ANSWER_PATH="data/model_predictions_converted"

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir="${RAW_ANSWER_PATH}/${MODEL_NAME}/${stage}_${group}"
output_file="${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${stage}_${group}.json"
echo ${output_file}
cmd="python -m toolbench.tooleval.convert_to_answer_format\
    --answer_dir ${answer_dir} \
    --method ${method} \
    --output ${output_file}"
echo $cmd
eval $cmd



export API_POOL_FILE=openai_key_mbz.json
# export API_POOL_FILE=openai_key.json
SAVE_PATH="data/results/pass_rate"
mkdir -p ${SAVE_PATH}
export EVAL_MODEL=gpt-4-turbo-2024-04-09
mkdir -p ${SAVE_PATH}/${MODEL_NAME}

cmd="python -m toolbench.tooleval.eval_pass_rate \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --reference_model ${MODEL_NAME} \
    --test_ids data/solvable_queries/test_query_ids \
    --max_eval_threads 3 \
    --evaluate_times 3 \
    --test_set ${stage}_${group} --overwrite"
echo $cmd
eval $cmd


export API_POOL_FILE=openai_key.json
SAVE_PATH="data/results/preference_rate"
PASS_RATE_PATH="data/results/pass_rate"
REFERENCE_MODEL=virtual-gpt35-16k-step16-cot
export EVAL_MODEL=gpt-4o
mkdir -p ${SAVE_PATH}

cmd="python -m toolbench.tooleval.eval_preference \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${MODEL_NAME} \
    --test_ids data/solvable_queries/test_query_ids/ \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --pass_rate_result_path ${PASS_RATE_PATH} \
    --max_eval_threads 3 \
    --use_pass_rate true \
    --evaluate_times 3 \
    --test_set ${stage}_${group} --overwrite"
echo $cmd
eval $cmd