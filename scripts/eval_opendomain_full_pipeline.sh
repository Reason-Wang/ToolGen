TOOLBENCH_KEY=""
OPENAI_KEY=""
export PYTHONPATH=./
export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"

# MODEL_NAME="virtual-gpt35-16k-step16-cot-opendomain"
# model_path="virtual-gpt35-16k-step16-cot"
# backbone_model="chatgpt_function"
# function_provider="retriever"

MODEL_NAME="ToolLlama-Llama-3-8B-t0.0-cot-toolretriever"
model_path="reasonwang/ToolLlama-Llama-3-8B"
function_provider="retriever"
backbone_model="toolchat"



# MODEL_NAME="ToolLlama-Llama-3-8B-t0.0-cot-toolretriever"
# model_path="reasonwang/ToolLlama-Llama-3-8B"
# function_provider="retriever"
# backbone_model="toolchat"


# MODEL_NAME="ToolLlama-v2-t0.0-cot-opendomain-toolretriever-retry-finish"
# model_path="ToolBench/ToolLLaMA-2-7b-v2"
# indexing="None"
# replace_file="None"
# function_provider="retriever"
# backbone_model="toolllama"



OUTPUT_DIR="data/answer/${MODEL_NAME}"
export CUDA_VISIBLE_DEVICES=0
stage="G2"
group="instruction"
method="CoT@1"


# Open domain setting
# corpus_tsv_path="data/retrieval/${stage}/corpus.tsv"
corpus_tsv_path="data/retrieval/corpus_G123.tsv"
# retrieval_model_path="reasonwang/BERT-${stage}"
retrieval_model_path="ToolBench/ToolBench_IR_bert_based_uncased"
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/${stage}_${group}

cmd="python toolbench/inference/qa_pipeline_multithread.py \
    --replace_file ${replace_file} \
    --model_path ${model_path} \
    --chatgpt_model ${GPT_MODEL} \
    --tool_root_dir data/toolenv/tools \
    --corpus_tsv_path ${corpus_tsv_path} \
    --retrieval_model_path ${retrieval_model_path} \
    --backbone_model ${backbone_model} \
    --openai_key ${OPENAI_KEY} \
    --max_observation_length 1024 \
    --method ${method} \
    --input_query_file data/solvable_queries/test_instruction/${stage}_${group}.json \
    --output_answer_file $OUTPUT_DIR/${stage}_${group} \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1 \
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