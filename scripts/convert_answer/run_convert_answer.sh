export RAW_ANSWER_PATH=data/answer
export CONVERTED_ANSWER_PATH=data/model_predictions_converted
export MODEL_NAME=llama-3-toolllama-g123-cot-opendomain
export test_set=G1_instruction
method="CoT@1"

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json
echo ${output_file}
python -m toolbench.tooleval.convert_to_answer_format\
    --answer_dir ${answer_dir} \
    --method ${method} \
    --output ${output_file}