import click
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoConfig
from src.zero_to_fp32 import get_optim_files, parse_optim_states, get_model_state_files, parse_model_states, \
    zero3_partitioned_param_info
import torch
import os
import gc
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from deepspeed.checkpoint.constants import (
    DS_VERSION,
    OPTIMIZER_STATE_DICT,
    SINGLE_PARTITION_OF_FP32_GROUPS,
    FP32_FLAT_GROUPS,
    ZERO_STAGE,
    PARTITION_COUNT,
    PARAM_SHAPES,
    BUFFER_NAMES,
    FROZEN_PARAM_SHAPES,
    FROZEN_PARAM_FRAGMENTS
)
debug = False


def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    if debug:
        for i in range(world_size):
            num_elem = sum(s.numel() for s in zero_model_states[i].frozen_param_fragments.values())
            print(f'rank {i}: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')

        frozen_param_shapes = zero_model_states[0].frozen_param_shapes
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in zero_model_states[0].frozen_param_fragments.values()]) * world_size
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(f'Frozen params: Need {wanted_numel} numels in {wanted_params} params')

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states, ds_checkpoint_dir, sharding_size=48):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = fp32_flat_groups[0].numel() * world_size
    # Reconstruction protocol: For zero3 we need to zip the partitions together at boundary of each
    # param, re-consolidating each param, while dealing with padding if any

    # merge list of dicts, preserving order
    param_shapes = {k: v for d in param_shapes for k, v in d.items()}

    if debug:
        for i in range(world_size):
            print(f"{FP32_FLAT_GROUPS}[{i}].shape={fp32_flat_groups[i].shape}")

        wanted_params = len(param_shapes)
        wanted_numel = sum(shape.numel() for shape in param_shapes.values())
        # not asserting if there is a mismatch due to possible padding
        avail_numel = fp32_flat_groups[0].numel() * world_size
        print(f"Trainable params: Have {avail_numel} numels to process.")
        print(f"Trainable params: Need {wanted_numel} numels in {wanted_params} params.")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    offset = 0
    total_numel = 0
    total_params = 0

    num_weights = len(param_shapes)
    partition_size = num_weights // sharding_size

    # Used to save state_dict to disk
    cache_dir = ds_checkpoint_dir + "/states"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for i, (name, shape) in enumerate(param_shapes.items()):
        print(f"Processing {name} {shape}")
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

        if debug:
            print(
                f"Trainable params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}"
            )

        # XXX: memory usage doubles here
        state_dict[name] = torch.cat(
            tuple(fp32_flat_groups[i].narrow(0, offset, partitioned_numel) for i in range(world_size)),
            0).narrow(0, 0, unpartitioned_numel).view(shape)
        offset += partitioned_numel


        # Save state_dict to disk
        # We do this to save memory
        if i % partition_size == partition_size - 1:
            # save state_dict to disk
            torch.save(state_dict, f'{cache_dir}/checkpoint_{i}.pt')
            state_dict = OrderedDict()
            gc.collect()

    if len(state_dict) > 0:
        torch.save(state_dict, f'{cache_dir}/checkpoint_{num_weights}.pt')
        state_dict = OrderedDict()
        gc.collect()

    offset *= world_size

    # Sanity check
    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")
    
    # Do not need fp32_flat_groups anymore
    del fp32_flat_groups
    gc.collect()

    # Reconstruct state_dict from disk
    # Load all cached states
    state_dict = OrderedDict()
    cache_dir = ds_checkpoint_dir + "/states"
    for file in os.listdir(cache_dir):
        state_dict.update(torch.load(os.path.join(cache_dir, file)))

    # recover shared parameters
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    # Do not need zero_model_states anymore
    del zero_model_states
    gc.collect()

    print(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


@click.command()
@click.option('--checkpoint_dir', type=str, help='Checkpoint directory')
@click.option('--pretrain_dir', type=str, help='Pretrain directory')
@click.option('--data_type', type=str, default='bfloat16', help='Data type')
def main(
    checkpoint_dir: str,
    pretrain_dir: str,
    data_type: str
):

    # checkpoint_dir = "data/checkpoints/65b_ckpts_hf_mini_stage2/iter_0013740_mix_sft"
    tag = None

    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir, data_type)
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    model_files = get_model_state_files(ds_checkpoint_dir)

    zero_model_states = parse_model_states(model_files)
    print(f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}')

    state_dict = OrderedDict()

    # buffers
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f"added {len(buffers)} buffers")

    _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states, ds_checkpoint_dir, sharding_size=24)

    config = AutoConfig.from_pretrained(pretrain_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
    if data_type == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif data_type == 'float16':
        torch_dtype = torch.float16
    elif data_type == 'float32':
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    model = AutoModelForCausalLM.from_pretrained(
        None,
        config=config,
        state_dict=state_dict,
        torch_dtype=torch_dtype
    )

    tokenizer.save_pretrained(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)

if __name__=="__main__":
    main()