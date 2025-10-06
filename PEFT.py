# %%
import os
import torch
from tqdm.auto import tqdm
import argparse
import numpy as np
import random
from datasets import Dataset

from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from trl import SFTTrainer
from trl import DPOTrainer, DPOConfig

from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

from concept_datasets import dataset_FT_funcs



# %%

def get_args(run_in_notebook = False):

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_index", type=int, default=1, help="Index of the model to use, \
                                                                    0:meta-llama/Llama-3.2-1B-Instruct, \
                                                                    1:meta-llama/Llama-3.1-8B-Instruct,  \
                                                                    2:meta-llama/Llama-3.1-70B-Instruct, \
                                                                    3:google/gemma-3-4b-it,              \
                                                                    4:google/gemma-3-12b-it,             \
                                                                    5:Qwen/Qwen3-8B,                     \
                                                                    6:google/gemma-2-2b-it,              \
                                                                    7:google/gemma-2-2b,                 \
                                                                    8:meta-llama/Llama-3.2-1B, \
                                                                    9:meta-llama/Llama-3.1-8B,  \
                                                                    10:meta-llama/Llama-3.1-70B, \
                                                                    11:google/gemma-3-4b-pt, \
                                                                    12:google/gemma-3-12b-pt, \
                                                                    13:Qwen/Qwen3-4B")
    parser.add_argument("--use_quantization",      action='store_true', help="Use quantization for the model", default=False)
    
    parser.add_argument("--dataset_source", type=str, default='carer_binary')
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--steps", type=int, default=20, help="Training Steps")

    parser.add_argument("--wd",   type=float, default=0.05, help="Weight Decay")
    parser.add_argument("--warmup", type=float, default=100, help="Warmup Steps")

    parser.add_argument("--alpha", type=int, default=100, help="Lora Alpha")
    parser.add_argument("--r",     type=int, default=32, help="Lora r")

    parser.add_argument("--method", type=str, default='DPO', choices = ['SFT', 'DPO'])
    
    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    args.device = "balanced" if torch.cuda.is_available() else "cpu"

    return args

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

args = get_args(run_in_notebook = in_notebook())
print(f"----------- Training {args.method} on {args.dataset_source}")

# %%
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


seed_everywhere(args.seed)

# %%
if args.model_index in [2, 4, 10]:
    args.use_quantization = True
    print('Automatically turning on quantization...')

if args.model_index in [3, 4, 11, 12]: # models that support vision
    UnslothModelClass = FastModel
else:
    UnslothModelClass = FastLanguageModel

# %%
UnslothModelClass

# %%
max_seq_length = 4096
def get_model(model_name, use_quantization, max_seq_length = max_seq_length):
    base_model, _ = UnslothModelClass.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = use_quantization,
    )

    auto_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
    )

    return base_model, auto_tokenizer #tokenizer

# %%
model_names =       ['unsloth/Llama-3.2-1B-Instruct', 'unsloth/Llama-3.1-8B-Instruct', 'unsloth/Meta-Llama-3.1-70B-Instruct', 
                     'unsloth/gemma-3-4b-it', 'unsloth/gemma-3-12b-it', 
                     'unsloth/Qwen3-8B', 
                     'unsloth/gemma-2-2b-it', 'unsloth/gemma-2-2b',
                     'unsloth/Llama-3.2-1B', 'unsloth/Llama-3.1-8B', 'unsloth/Llama-3.1-70B',
                     'unsloth/gemma-3-4b-pt', 'unsloth/gemma-3-12b-pt', 
                     'unsloth/Qwen3-4B'
                     ]

model_short_names = ['Llama3.2_1B', 'Llama3.1_8B', 'Llama3.1_70B', 
                     'Gemma3_4B', 'Gemma3_12B', 
                     'Qwen3_8B', 
                     'Gemma2_2B', 'Gemma2_2B_PT', 
                     'Llama3.2_1B_PT', 'Llama3.1_8B_PT', 'Llama3.1_70B_PT', 
                     'Gemma3_4B_PT', 'Gemma3_12B_PT',
                     'Qwen3_4B']


model_name, model_short_name = list(zip(model_names, model_short_names))[args.model_index]    
if args.use_quantization:    
    model_name = model_name + "-bnb-4bit"
    model_short_name = model_short_name + "_quantized"

base_model, tokenizer = get_model(model_name, args.use_quantization)

# %%
all_data, labels, concepts, dataset_type = dataset_FT_funcs[args.dataset_source]()

# %%
assert dataset_type == 'multitask'
for task_labels in labels:
    assert task_labels.shape[1] == 1
    for c in range(len(task_labels[0])):
        assert sorted(np.unique(task_labels[:, c])) == [0, 1], f"In {args.vec_type} mode, labels should be binary, but found {np.unique(task_labels[:, c])} for concept {c} in task {task_labels}"

# %%
if args.method == 'SFT': # only keep the samples with + label
    new_all_data = []
    for t in range(len(labels)):
        new_all_data.append([])
        for r in range(len(labels[t])):
            if labels[t][r] == 1:
                new_all_data[-1].append(all_data[t][r])
    all_data = new_all_data
    del labels
elif args.method == 'DPO': # check 
    new_all_data = []
    for t in range(len(labels)):
        new_all_data.append([])
        for r in range(len(labels[t]) // 2):
            assert labels[t][2 * r] == 1 and labels[t][2 * r + 1] == 0, f"In DPO mode, should be pairs of (1, 0) but found {labels[t][2 * r]} and {labels[t][2 * r + 1]} for task {t} and pair {r}"
            row1 = all_data[t][2 * r]
            row2 = all_data[t][2 * r + 1]
            assert row1['system'] == row2['system'], f'In DPO mode, the system and user messages should be the same for both rows, but found system message {row1["system"]} AND {row2["system"]} for task {t} and pair {r}'
            assert row1['user'] == row2['user'], f'In DPO mode, the system and user messages should be the same for both rows, but found user message {row1["user"]} AND {row2["user"]} for task {t} and pair {r}'
            new_all_data[-1].append({'system': row1['system'], 'user': row1['user'], 'chosen': row1['assistant'], 'rejected': row2['assistant']})
    all_data = new_all_data
    del labels
else:
    raise ValueError(f"Method {args.method} not supported. Choose from 'SFT' or 'DPO'.")
    

# %%
all_data[0][0]

# %%
concept_to_id = {concept: i for i, concept in enumerate(concepts)}
id_to_concept = {i: concept for i, concept in enumerate(concepts)}

def apply_chat_template_for_SFT(data, tokenizer):
    new_data = []
    for i, d in enumerate(data):
        conversation = [{"role": "system", "content": d['system']},
                {"role": "user", "content": d['user']},
                {"role": "assistant", "content": d['assistant']},]
        t = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = False, enable_thinking=False)
        
        if tokenizer.bos_token:
                t = t.replace(tokenizer.bos_token, '')

        new_data.append({'conversation': conversation, 'text': t})
    return new_data

def apply_chat_template_for_DPO(data, tokenizer):
    new_data = []
    for i, d in enumerate(data):
        conversation_base = [{"role": "system", "content": d['system']},
                {"role": "user", "content": d['user']},]
        t_base = tokenizer.apply_chat_template(conversation_base, tokenize = False, add_generation_prompt = True, enable_thinking=False)
        if tokenizer.bos_token:
                t_base = t_base.replace(tokenizer.bos_token, '')

        conversation_c = [{"role": "system", "content": d['system']},
                {"role": "user", "content": d['user']},
                {"role": "assistant", "content": d['chosen']},]
        t_c = tokenizer.apply_chat_template(conversation_c, tokenize = False, add_generation_prompt = False, enable_thinking=False)
        if tokenizer.bos_token:
                t_c = t_c.replace(tokenizer.bos_token, '')        
        t_c = t_c.replace(t_base, '')

        conversation_r = [{"role": "system", "content": d['system']},
                {"role": "user", "content": d['user']},
                {"role": "assistant", "content": d['rejected']},]
        t_r = tokenizer.apply_chat_template(conversation_r, tokenize = False, add_generation_prompt = False, enable_thinking=False)
        if tokenizer.bos_token:
                t_r = t_r.replace(tokenizer.bos_token, '')                
        t_r = t_r.replace(t_base, '')


        new_data.append({'prompt': t_base, 'chosen': t_c, 'rejected': t_r})
    return new_data    

# %%
# torch._dynamo.config.cache_size_limit = 32

# %%
import gc

for c, concept in enumerate(concepts):
    try:
        del base_model
        del tokenizer
        del model
        del train_dataset
        del dataset
        del trainer
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    base_model, tokenizer = get_model(model_name, args.use_quantization)

    print(f'-------------------- training concept {concept} ------------------------------')
    data = all_data[c]

    if args.method == 'SFT':
        train_dataset = apply_chat_template_for_SFT(data, tokenizer)
        print('--------------------- sample train text: -----------------------------')
        print(train_dataset[0]['text'])        
    elif args.method == 'DPO':
        train_dataset = apply_chat_template_for_DPO(data, tokenizer)
        print('--------------------- sample train text: -----------------------------')
        print('Prompt:', train_dataset[0]['prompt'])
        print('Chosen:', train_dataset[0]['chosen'])
        print('Rejected:', train_dataset[0]['rejected'])

    dataset = Dataset.from_list(train_dataset)
    print('dataset stats:', dataset)


    model = UnslothModelClass.get_peft_model(
        base_model,

        # finetune_vision_layers     = False, # Turn off for just text!
        # finetune_language_layers   = True,  # Should leave on!
        # finetune_attention_modules = True,  # Attention good for GRPO
        # finetune_mlp_modules       = True,  # SHould leave on always!

        r = args.r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = args.seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    if args.method == 'SFT':
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
            dataset_num_proc = 4,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 8,
                gradient_accumulation_steps = 2,
                warmup_steps = args.warmup,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = args.steps,
                learning_rate = args.lr,
                fp16 = (not is_bfloat16_supported()) and args.use_quantization,
                bf16 = is_bfloat16_supported() and args.use_quantization,
                logging_steps = 5,
                optim = "adamw_8bit",
                weight_decay = args.wd,
                lr_scheduler_type = "linear",
                seed = args.seed,
                output_dir = "outputs",
                overwrite_output_dir = True,
                report_to = "none", # Use this for WandB etc
            ),
        )

        if 'Llama-3' in model_name:
            trainer = train_on_responses_only(
                trainer,
                instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
        elif 'gemma-3' in model_name:
            trainer = train_on_responses_only(
                trainer,
                instruction_part = "<start_of_turn>user\n",
                response_part = "<start_of_turn>model\n",
            )
        elif 'Qwen3' in model_name:
            trainer = train_on_responses_only(
                trainer,
                instruction_part = "<|im_start|>user\n",
                response_part = "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        else:
            assert False, 'not implemented'



                
    elif args.method == 'DPO':
        trainer = DPOTrainer(
            model = model,
            ref_model = None,
            tokenizer = tokenizer,
            train_dataset = dataset,
            max_prompt_length = max_seq_length,
            beta = 0.1,
            
            
            max_length = 1024,
            
            args = DPOConfig(
                per_device_train_batch_size = 8,
                gradient_accumulation_steps = 2,
                # warmup_ratio = 0.1,
                warmup_steps = args.warmup,
                # num_train_epochs = 1,
                max_steps = args.steps,
                learning_rate = args.lr,
                fp16 = (not is_bfloat16_supported()) and args.use_quantization,
                bf16 = is_bfloat16_supported() and args.use_quantization,
                logging_steps = 5,
                optim = "adamw_8bit",
                weight_decay = args.wd,
                lr_scheduler_type = "linear",
                seed = args.seed,
                output_dir = "outputs",
                overwrite_output_dir = True,
                report_to = "none", # Use this for WandB etc
                ),
        )   



    
    print("--------- Starting training ----------------")
    trainer_stats = trainer.train()

    print("---------------- Saving model ------------------")
    args.dataset_source_with_count = args.dataset_source
    
    if args.seed != 42:
        args.dataset_source_with_count += f'_seed{args.seed}'

    save_path = f"PEFT_models/{args.method}/{model_short_name}/{args.lr}_{args.steps}/{args.dataset_source_with_count}/{concept}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print('model saved to: ', save_path)
    print(f'####################################### end of concept {concept} #######################################')
