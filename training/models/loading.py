from utils.distributed import is_main_process
import transformers
import torch
from unidecode import unidecode


def load_tokenizer(model_name_or_path, cache_dir=None, virtual_tokens=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    if virtual_tokens:
        # if "llama-3" in model_name_or_path.lower():
        #     tokenizer = transformers.AutoTokenizer.from_pretrained(
        #         "meta-llama/Meta-Llama-3-8B", 
        #         cache_dir=cache_dir,
        #     )
        # else:
        #     raise ValueError(f"Virtual tokens not supported for tokenizer {model_name_or_path}")
        with open('src/configs/virtual_tokens.txt', 'r') as f:
            virtual_tokens = f.readlines()
            virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
        tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
        if is_main_process():
            print(f"Added {len(virtual_tokens)} virtual tokens")

    return tokenizer


def load_model(model_name_or_path, architecture, tokenizer=None, flash_attention=False, cache_dir=None, virtual_tokens=False):
    if architecture == 'causal':
        # Check hf_home
        # rank = get_rank()
        # print(f"Rank {rank}: {os.environ['HF_HOME']}")
        # print(f"Rank {rank}: cache dir: {args.cache_dir}")
        if flash_attention:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2'
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
            )
    elif architecture == 'seq2seq':
        if flash_attention:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation='flash_attention_2'
            )
        else:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
            )
    else:
        raise ValueError(f"Architecture {architecture} not supported")
    
    if virtual_tokens:
        model.resize_token_embeddings(len(tokenizer))
        if is_main_process():
            print(f"Model resized token embeddings to {len(tokenizer)}")

        with open('src/configs/virtual_tokens.txt', 'r') as f:
            virtual_tokens = f.readlines()
            virtual_tokens = [unidecode(vt).strip() for vt in virtual_tokens]
        combined_tokens = []
        for vt in virtual_tokens:
            combined_token = vt[2:-2].split("&&")
            combined_tokens.append(combined_token)
            
        for combined_token, virtual_token in zip(combined_tokens, virtual_tokens):
            combined_token_ids = tokenizer(" ".join(combined_token), add_special_tokens=False).input_ids
            virtual_token_id = tokenizer(virtual_token, add_special_tokens=False).input_ids
            # print(combined_token_ids)
            # print(virtual_token_id)
            assert len(virtual_token_id) == 1
            # print(model.device)
            combined_token_embeddings = model.model.embed_tokens(torch.tensor(combined_token_ids).to(model.device))
            # print(combined_token_embeddings.shape)
            embedding = torch.mean(combined_token_embeddings, dim=0)
            # print(embedding.shape)
            model.model.embed_tokens.weight.data[virtual_token_id[0]] = embedding
    else:
        if is_main_process():
            print(f"Initialized from {model_name_or_path} without adding embeddings.")
    
    return model
