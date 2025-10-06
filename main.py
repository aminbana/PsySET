
# %%
import argparse

from tqdm.auto import tqdm

import os
import time
import random

from prompt_manager import get_prompt
from path_manager import get_main_path
import gc



# %%


def get_args(run_in_notebook=False):
    parser = argparse.ArgumentParser(description='Run the LLM')

    # Model arguments
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
    
    parser.add_argument("--use_quantization", action='store_true', help="Use quantization for the model", default=False)
    # Task arguments
    parser.add_argument('--dataset', type=str, default='emotion_offline', help='Evaluation dataset to use')

    parser.add_argument('--result_save_path', type=str, default='results/', required=False, help='Path to save the results')
    parser.add_argument('--skip_with_file', type=str, default=None)
    # Inference arguments
    parser.add_argument('--bs', type=int, default=4, required=False, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.6, required=False, help='Temperature for sampling')
    parser.add_argument('--argmax', action='store_true', help='Use argmax for sampling instead of sampling from the distribution', default=False)
    parser.add_argument('--use_kv_cache', action='store_true', help='Use kv cache for faster inference', default=True)
    parser.add_argument('--eval_only', action='store_true', help='Eval only', default=False)
    
    parser.add_argument('--steer_type', type=str, default='None', choices=['None', 'Intervention', 'Prompt', 'SFT', 'DPO'], help='Type of steering to use')
    
    # General arguments
    parser.add_argument('--concept_source', type=str, default='goemotions_instruct_full_binary')
    parser.add_argument('--concept', type=str, default='fear', help='Concept to steer towards')
    
    # Intervention arguments
    parser.add_argument('--intervention_type', type=str, default='add', choices=['add', 'replace'], help='Type of steering to use')
    parser.add_argument('--intervention_source', type=str, default='probeall', choices=['probe', 'probeassistant', 'meandiff', 'meandiffall', 'meandiffassistant', 'mean', 'probeall'], help='Source of the steering signal')
    parser.add_argument('--steer_coeff', type=float, default= 5.0, required=False, help='Coefficient for intervention')
    parser.add_argument('--steer_layers', type=str, default='all')
    parser.add_argument('--steer_locs', type=str, default='7')
    parser.add_argument('--steer_normalize', type=bool, default=True, help='Whether to normalize the steering vector before applying the coefficient')
    parser.add_argument('--steer_renormalize', type=bool, default=False, help='Whether to rescale the manipulated activation to the original range')
    
    # Prompt arguments
    parser.add_argument('--prompt_strength', type=str, default='medium', choices=['high', 'low', 'medium', 'neg', 'neg_high', 'neg_low', 'very_high', 'very_low'], help='Strength of the prompt steering')
    parser.add_argument('--prompt_template', type=str, default='template_1', help='Template to use for the prompt steering')
    parser.add_argument('--prompt_method'  , type=str, default='p2', choices=['zero', 'few', 'p2'], help='Strength of the prompt steering')

    # SFT and DPO arguments
    parser.add_argument('--PEFT_lr', type=float, default=1e-4)
    parser.add_argument('--PEFT_steps', type=int, default=1024)
    
    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    
    args.device = 'cuda'
    
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

# %%
if args.model_index in [2, 4, 10]:
    args.use_quantization = True
    print('Automatically turning on quantization...')

# %%
if args.argmax:
    print("Using argmax sampling, setting temperature to 0.6")
    args.temperature = 0.6

# %%
print(f'Dataset: {args.dataset}')
print(f'Batch size: {args.bs}')
print(f'Argmax or Sample: {"argmax" if args.argmax else "sample"}')
print(f'Temperature:', args.temperature)
print(f'Steer type: {args.steer_type}')
print(f'device: {args.device}')
if args.steer_type != 'None':
    print(f'Concept: {args.concept}')
    if args.steer_type == 'Intervention':
        print(f'Intervention source: {args.intervention_source}')
        print(f'Steer coeff: {args.steer_coeff}')
        print(f'Intervention layers: {args.steer_layers}')
    elif args.steer_type == 'Prompt':
        print(f'Prompt Strength: {args.prompt_strength}')
        print(f'Prompt Template: {args.prompt_template}')
        print(f'Prompt Method: {args.prompt_method}')
    elif args.steer_type == 'SFT':
        print(f'SFT lr: {args.PEFT_lr}, steps: {args.PEFT_steps}')
    elif args.steer_type == 'DPO':
        print(f'DPO lr: {args.PEFT_lr}, steps: {args.PEFT_steps}')

    


# %%
model_names =       ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-70B-Instruct', 
                     'google/gemma-3-4b-it', 'google/gemma-3-12b-it', 
                     'Qwen/Qwen3-8B', 
                     'google/gemma-2-2b-it', 'google/gemma-2-2b',
                     'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-70B',
                     'google/gemma-3-4b-pt', 'google/gemma-3-12b-pt', 
                     'Qwen/Qwen3-4B'
                     ]

model_short_names = ['Llama3.2_1B', 'Llama3.1_8B', 'Llama3.1_70B', 
                     'Gemma3_4B', 'Gemma3_12B', 
                     'Qwen3_8B', 
                     'Gemma2_2B', 'Gemma2_2B_PT', 
                     'Llama3.2_1B_PT', 'Llama3.1_8B_PT', 'Llama3.1_70B_PT', 
                     'Gemma3_4B_PT', 'Gemma3_12B_PT',
                     'Qwen3_4B']

model_name, model_short_name = list(zip(model_names, model_short_names))[args.model_index]  
print('model_name:', model_name, args.model_index)

if args.use_quantization:
    model_short_name = model_short_name + "_quantized"

save_dir = get_main_path(args.temperature, args.argmax, args.result_save_path, model_short_name, args.dataset, args.steer_type, args.concept_source, args.concept, args.steer_layers, args.intervention_type, args.intervention_source, args.steer_coeff, args.prompt_method, args.prompt_template, args.prompt_strength, args.PEFT_lr, args.PEFT_steps)

if args.skip_with_file:
    file_path = save_dir + '/' + args.skip_with_file
    if os.path.exists(file_path):
        print('File path exists, exiting the run:', file_path)
        assert False, 'Exit...'

# %%


# %%
import torch
import numpy as np
from unsloth import FastLanguageModel, FastModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from steer_manager import get_steer_fn, load_steer_vectors, apply_wieghted_sum, get_default_steer_fn
import torch.nn.functional as F
import pandas as pd
def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

# %%
set_seed_everywhere(args.seed)


from LLMs.my_llama import SteeredLlamaForCausalLM
from LLMs.my_gemma3 import SteeredGemma3ForCausalLM
from LLMs.my_qwen3 import SteeredQwen3ForCausalLM
from LLMs.my_gemma2 import SteeredGemma2ForCausalLM
from transformers import BitsAndBytesConfig

model_classes =     [SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, 
                     SteeredGemma3ForCausalLM, SteeredGemma3ForCausalLM, 
                     SteeredQwen3ForCausalLM, 
                     SteeredGemma2ForCausalLM, SteeredGemma2ForCausalLM,
                     SteeredLlamaForCausalLM, SteeredLlamaForCausalLM, SteeredLlamaForCausalLM,
                     SteeredGemma3ForCausalLM, SteeredGemma3ForCausalLM,
                     SteeredQwen3ForCausalLM]

model_class = model_classes[args.model_index]    

print('model:', args.model_index, model_class, model_name, model_short_name)

# %%
if args.model_index in [3, 4, 11, 12]: # models that support vision
    UnslothModelClass = FastModel
else:
    UnslothModelClass = FastLanguageModel

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token    

if args.steer_type in ['SFT', 'DPO']:
    
    max_seq_length = 4096
    load_path = f"PEFT_models/{args.steer_type}/{model_short_name}/{args.PEFT_lr}_{args.PEFT_steps}/{args.concept_source}/{args.concept}"

    model, _tok = UnslothModelClass.from_pretrained(
        model_name = load_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = args.use_quantization,
    )
    model.eval()
    UnslothModelClass.for_inference(model) # Enable native 2x faster inference
    
else:
    if args.use_quantization:    
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
    else:
        quantization_config = None

    try:
        model.eval()
    except:
        model = model_class.from_pretrained(model_name, device_map=args.device, quantization_config=quantization_config) 
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    torch.cuda.empty_cache()


# %%
model_name

# %%
if args.steer_type == 'Intervention':
    steer_vecs, steer_layers, steer_locs, steer_tokens = load_steer_vectors(model_short_name, model.config.num_hidden_layers, model.config.hidden_size, [args.concept], [args.concept_source], args.steer_layers, args.steer_locs, args.intervention_source, args.steer_normalize)
    steer_vec = apply_wieghted_sum(steer_vecs, torch.tensor([args.steer_coeff]), model.device)
    steer_fn = get_steer_fn(steer_vec, steer_layers, steer_locs, steer_tokens, model.config.hidden_size, args.intervention_type, args.steer_renormalize)
    system_prompt_suffix = ""

    model.set_steer_fn(steer_fn)

elif args.steer_type == 'None':
    steer_fn = get_default_steer_fn()
    system_prompt_suffix = ""
    model.set_steer_fn(steer_fn)
    
elif args.steer_type == 'Prompt':
    steer_fn = get_default_steer_fn()
    model.set_steer_fn(steer_fn)
    system_prompt_suffix = get_prompt(domain_name = args.concept_source, prompt_method=args.prompt_method, template_name=args.prompt_template, concept=args.concept, prompt_strength=args.prompt_strength)

elif args.steer_type in ['SFT', 'DPO']:
    system_prompt_suffix = ""

else:
    raise ValueError(f"Not implemented steer type {args.steer_type}")

# %%


os.makedirs(save_dir, exist_ok=True)    
print('created directory:', save_dir)



# %%
import itertools

class  SimpleDataset:
    def __init__(self, eval_only = False):
        self.eval_only = eval_only
        self.results = None
        self.description = 'Pretend that you are a human. How would you feel today?' #Generate some synthetic reviews with a positive sentiment. Use the following template:\n\ntext:[review1]\ntext:[review2]\n\nand so on.
        self.done = False
    
    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return False
        
    def get_system_prompt(self, preffered_batch_size):
        return [''] * 10

    def get_user_prompt(self, preffered_batch_size):
        self.done = True
        return [self.description] * 10
    
    def get_assistant_prompt(self, preffered_batch_size):
        return [''] * 10
    
    def is_finished(self):
        return self.done or self.eval_only
    
    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            llm_generations[i] = llm_generations[i]
        self.result = llm_generations

        self.done = True
        
    def finalize(self, save_path = None, concept = None):
        for r in self.result:
            print(r)
            print('--------------------------------')
                
    def get_unique_name(self):
        return 'simple'
    
    def get_max_len(self):
        return 256

    def get_class_labels(self):
        return [' I', ' you'], 0, 5
    
    def get_progress(self):
        return 1


# Mapping dataset names to classes
from my_datasets.truthfulness import TruthfulnessDataset
from my_datasets.safety import SafetyDataset
from my_datasets.fairness import FairnessDataset
from my_datasets.privacy import PrivacyDataset
from my_datasets.robustness import RobustnessDataset
from my_datasets.ethics import EthicsDataset
from my_datasets.trait_eval import PersonalityEvaluator
from my_datasets.emotion_eval import EmotionDataset

# decision
from functools import partial

dataset_map = {
    'simple': SimpleDataset,
    'truthfulness': TruthfulnessDataset,
    'safety': SafetyDataset,
    'privacy': PrivacyDataset,
    'fairness': FairnessDataset,
    'robustness': RobustnessDataset,
    'ethics': EthicsDataset,
    
    'personality': partial(PersonalityEvaluator, gpt_eval_model='online'),
    'personality_offline': partial(PersonalityEvaluator, gpt_eval_model='offline'),
    
    'emotion': partial(EmotionDataset, gpt_eval_model='online'),
    'emotion_offline': partial(EmotionDataset, gpt_eval_model='offline'),
    

}

# %%

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("The model has a chat template configured.")
    # print(f"Chat template: {tokenizer.chat_template}")
    chat_enabled = True
else:
    chat_enabled = False
    print("The model does not have an explicit chat template configured.")

# %%
if chat_enabled:
    messages_with_system = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
        ]

    try:
        # Attempt to apply the template with a system message
        
        formatted_input = tokenizer.apply_chat_template(messages_with_system, tokenize=False)
        print("System role is likely supported.")
        system_role_supported = True
        # You can also inspect `tokenizer.chat_template` if it's explicitly defined
        # print(tokenizer.chat_template)
    except Exception as e:
        
        if "System role not supported" in str(e):
            print("System role is not supported by this model's tokenizer.")
            system_role_supported = False
        else:
            print(f"An error occurred: {e}")


# %%

dataset = dataset_map[args.dataset](eval_only=args.eval_only)
last_progress = 0
max_tokenized_len = 0

with tqdm(total=100, unit="iteration", desc=f"Running dataset {dataset.get_unique_name()}") as pbar:
    while not dataset.is_finished():
            
        user_prompts = dataset.get_user_prompt(args.bs)
        system_prompts = dataset.get_system_prompt(args.bs)
        assistant_prompts = dataset.get_assistant_prompt(args.bs)
        
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        
        if not isinstance(system_prompts, list):
            system_prompts = [system_prompts]
        
        if not isinstance(assistant_prompts, list):
            assistant_prompts = [assistant_prompts]
        
        assert len(user_prompts) == len(system_prompts) == len(assistant_prompts), f"User prompts: {len(user_prompts)}, System prompts: {len(system_prompts)}, Assistant prompts: {len(assistant_prompts)}"
            
        
        new_prompts = []
        for user_prompt, system_prompt, assistant_prompt in zip(user_prompts, system_prompts, assistant_prompts):
            if system_prompt != '':
                system = system_prompt + ' ' + system_prompt_suffix
            else:
                system = system_prompt_suffix

            if chat_enabled:
                if system_role_supported:
                    chat = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': user_prompt},
                            # {'role': 'assistant', 'content': assistant_prompt}
                            ]
                else:
                    system_ = f'{system}\n\n' if system != '' else ''
                    chat = [
                            {'role': 'user', 'content': system_ + user_prompt},
                            # {'role': 'assistant', 'content': assistant_prompt}
                            ]                        
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking = False) + assistant_prompt
            
                # removing bos token from the prompt because it is added by the tokenizer again in the future tokenize call            
                bos_token = tokenizer.bos_token
                if tokenizer.bos_token:
                    prompt = prompt.replace(tokenizer.bos_token, '')
            else:
                if system != '':
                    system += '\n\n'
                if assistant_prompt != '':
                    assistant_prompt = '\n\n' + assistant_prompt
                
                prompt = system + user_prompt + assistant_prompt
            new_prompts.append(prompt)
        
        prompts = new_prompts

            
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        len_original_prompts = [len(x) for x in inputs.input_ids]
        
        generation = model.generate(**inputs, max_new_tokens=dataset.get_max_len(), do_sample = not args.argmax, temperature=args.temperature, 
                                    return_dict_in_generate=True, output_scores=True, use_cache=args.use_kv_cache, disable_compile=True)
        
        cropped_generations = []
        full_response = []
        for b in range(inputs.input_ids.shape[0]):
            cropped_generations.append(tokenizer.decode(generation.sequences[b, len_original_prompts[b]:], skip_special_tokens=True))
            full_response.append(tokenizer.decode(generation.sequences[b, :], skip_special_tokens=False))
        
        
        ########## Extracting logprobs and topk tokens
        topk_tokens = []
        topk_logprobs = []        
        
        class_labels, t0, k = dataset.get_class_labels()
        
        generation_scores = generation.scores[t0]
        for b in range(inputs.input_ids.shape[0]):
            logprobs = F.log_softmax(generation_scores[b], dim=-1)
            topk = torch.topk(logprobs, k, dim=-1)
            topk_tokens.append(tokenizer.batch_decode(topk.indices.cpu(), skip_special_tokens=False))
            topk_logprobs.append(topk.values.cpu().tolist())

        tokenized_class_labels = []
        for c in class_labels:
            t = tokenizer.encode(c, add_special_tokens=False, return_tensors='pt')
            if t.shape[1] > 1:
                print(f"Warning: Class label {c} is longer than 1 token. Using only the first token.")
            t = t[:, 0]
            tokenized_class_labels.append(t.cpu().tolist())

        logprobs = torch.log_softmax(generation_scores, dim=-1)
        class_label_logprobs = logprobs[:, tokenized_class_labels].cpu().tolist()
        
        dataset.process_results(llm_generations=cropped_generations, full_prompt=full_response, topk_tokens=topk_tokens, topk_logprobs=topk_logprobs, target_logprobs=class_label_logprobs)
        torch.cuda.empty_cache()
        
        pbar.update(100 * (dataset.get_progress() - last_progress))
        last_progress = dataset.get_progress()

    pbar.update((dataset.get_progress() - last_progress) == 1.0)

if dataset.get_clean_gpu_memory_before_finalize():
    try:
        del model, inputs, generation, tokenizer
        del generation_scores, logprobs, topk, topk_logprobs, class_label_logprobs
        del steer_vecs, steer_vec, steer_fn
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

dataset.finalize(save_path=save_dir)

