export OPENAI_KEY="Set your openai api key here"
export API_POOL_FILE=openai_keys.json
export CONVERTED_ANSWER_PATH=data/model_predictions_converted
export SAVE_PATH=data/results/preference_results
export PASS_RATE_PATH=data/results/pass_rate
export REFERENCE_MODEL=virtual-gpt35-16k-step16-cot
export CANDIDATE_MODEL=test
export EVAL_MODEL=gpt-4o-2024-05-13
test_set="G2_instruction"
mkdir -p ${SAVE_PATH}


cmd="python -m evaluation.toolbench.tooleval.eval_preference \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids data/solvable_queries/test_query_ids/ \
    --save_path ${SAVE_PATH} \
    --pass_rate_result_path ${PASS_RATE_PATH} \
    --max_eval_threads 3 \
    --use_pass_rate true \
    --evaluate_times 3 \
    --test_set ${test_set}"
echo $cmd
eval $cmd