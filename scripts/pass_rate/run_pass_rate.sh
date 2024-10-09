# cd  toolbench/tooleval
export API_POOL_FILE=openai_key.json
# export OPENAI_API_BASE="https://api.openai.com/v1" 
export CONVERTED_ANSWER_PATH=data/model_predictions_converted
export SAVE_PATH=data/results/pass_rate
mkdir -p ${SAVE_PATH}
# export CANDIDATE_MODEL="llama-3-toolllama-g123"
# export CANDIDATE_MODEL="vagent-llama-3-toolllama-g123"
# export CANDIDATE_MODEL="vagent-llama-3-toolllama-g123-serverfix"
# CANDIDATE_MODEL="llama-3-toolllama-g123-serverfix"
export CANDIDATE_MODEL=virtual_gpt35_16k_cot
# export CANDIDATE_MODEL=llama-3-toolllama-g123-cot-opendomain
# export CANDIDATE_MODEL=vagent-llama-3-toolllama-g123-cot-samplefalse
# export CANDIDATE_MODEL=vagent-llama-3-toolllama-g123-cot
# CANDIDATE_MODEL=vagent-llama-3-toolllama-g123-wplanning-step16-cot
# CANDIDATE_MODEL=vagent-llama-3-toolllama-g123-wplanning-wretrieval-cot
# CANDIDATE_MODEL=llama-3-toolllama-g123-cot
TEST_SET="G1_instruction"
export EVAL_MODEL=gpt-4-turbo-preview
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

python -m toolbench.tooleval.eval_pass_rate \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids data/solvable_queries/test_query_ids \
    --max_eval_threads 3 \
    --evaluate_times 3 \
    --test_set ${TEST_SET} --overwrite