# %%
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from steer_vec_utils import (TextDataset, probe_classification, extract_hidden_states, probe_regression, seed_everywhere)
import argparse
from concept_datasets import dataset_funcs
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from copy import deepcopy


# %%
MAX_SAMPLES_DEFAULT=800
def get_args(run_in_notebook = False):

    parser = argparse.ArgumentParser()

    parser.add_argument("--bs", type=int, default=2, help="Model forward batch size")
    parser.add_argument("--model_index", type=int, default=11, help="Index of the model to use, \
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
    
    parser.add_argument("--dataset_source", type=str, default='goemotions_instruct_full_binary')
    parser.add_argument("--target_tokens", type=str, default='all', choices=['last', 'all', 'assistant'], help="Type of vector to use for the concept; meandiff only applies to last token, meandiffall applies to all tokens, meandiffassistant applies to all tokens in assistant part")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES_DEFAULT, help="Max number of samples to use from the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default='steer_vectors')
    
    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    args.device = "auto" if torch.cuda.is_available() else "cpu"

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
seed_everywhere(args.seed)

# %%
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

model_name, model_short_name, model_class = list(zip(model_names, model_short_names, model_classes))[args.model_index]


if 'gemma' in model_name:
    assistant_tag = '<start_of_turn>model\n'
elif 'llama' in model_name:
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>\n\n'    
elif 'Qwen' in model_name:
    assistant_tag = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
else:
    raise ValueError(f"Model {model_name} is not supported")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.use_quantization:    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_short_name = model_short_name + "_quantized"
else:
    quantization_config = None

try:
    model.eval()
except:
    model = model_class.from_pretrained(model_name, device_map=args.device, quantization_config=quantization_config)
    model.eval()

# %%
data_raw, labels_raw, concepts, dataset_type = dataset_funcs[args.dataset_source]()
assert dataset_type in ['classification-multiclass', 'multitask', 'classification-multilabel'], 'not implemented yet'

# %%
concepts_to_ids = {concept: i for i, concept in enumerate(concepts)}
ids_to_concepts = {i: concept for i, concept in enumerate(concepts)}
print('Contepts are:', concepts)

# %%

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("The model has a chat template configured.")
    # print(f"Chat template: {tokenizer.chat_template}")
    chat_enabled = True
else:
    chat_enabled = False
    print("The model does not have an explicit chat template configured.")

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


if 'assistant' in args.target_tokens:
    assert chat_enabled, "The model does not support chat format required for assistant target_tokens"

# %%
def apply_chat_template(data, tokenizer):
    all_data = []
    for task in data:
        this_data = []
        for d in task:
            chat = []
            if chat_enabled:
                if system_role_supported:
                    if 'system' in d and d['system'] != '':
                        chat.append({'role': 'system', 'content': d['system']})
                    
                    chat.append({'role': 'user', 'content': d['user']})
                else:
                    system_ = f'{d["system"]}\n\n' if d["system"] != '' else ''
                    chat.append({'role': 'user', 'content': system_ + d['user']})

                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking = False)

                if 'assistant' in d:
                    chat += d['assistant']

                if tokenizer.bos_token:
                    chat = chat.replace(tokenizer.bos_token, '')
            else:
                system = d['system'] if 'system' in d else ''
                if system != '':
                    system += '\n\n'
                assistant_prompt = d['assistant'] if 'assistant' in d else ''
                if assistant_prompt != '':
                    assistant_prompt = '\n\n' + assistant_prompt

                chat = system + d['user'] + assistant_prompt
            this_data.append(chat)
        all_data.append(this_data)  
    return all_data

def sample_data(all_data, all_labels, max_samples):
    if max_samples > 0:
        data = []
        labels = []
        for t, task in enumerate(all_data):
            indices = torch.randperm(len(task))[:int(max_samples)]
            task = [task[i] for i in indices]        
            data.append(task)
            labels.append(all_labels[t][indices])
        
    else:
        data = all_data
        labels = all_labels

    return data, labels

# %%
if 'assistant' in args.target_tokens:
    for task in data_raw:
        for d in task:
            assert len(d['assistant']) > 0, f"In {args.target_tokens} tokens mode, assistant part should not be empty, but found empty in {d}"

# %%
print(f"Original Dataset size: {[len(task) for task in data_raw]}")

sampled_data, labels = sample_data(data_raw, labels_raw, args.max_samples)
data = apply_chat_template(sampled_data, tokenizer)

print(f"Dataset size after sampling: {[len(task) for task in data]}")


# %%
idx = 0
print(data[0][idx])
print(labels[0][idx])

# %%
possible_vec_types = ['probe', 'meandiff']

labels_set = []
for task_labels in labels:
    labels_set.append([])
    for c in range(len(task_labels[0])):
        labels_set[-1].append(sorted(np.unique(task_labels[:, c]).tolist()))
        assert labels_set[-1][-1] != [0, 1] or dataset_type != 'classification-multiclass', 'use classification-multilabel when the labels are binary'
        if 'meandiff' in possible_vec_types:
            if labels_set[-1][-1] != [0, 1]:
                print('Excluding meandiff as the labels are not binary')
                possible_vec_types.remove('meandiff')
    


# %%
print("Extracting vectors for concept dataset:", args.dataset_source, 'with vector types:', possible_vec_types, 'from token:', args.target_tokens, 'using model index:', args.model_index, 'and batch size:', args.bs)

# print("Sample instance:")
# for i, task in enumerate(data):
#     print(f"Task {i}: {task[0]}")
#     print(f"Label {i}: {labels[i][1]}")
#     print()
#     print(f"Task {i}: {task[-1]}")
#     print(f"Label {i}: {labels[i][-1]}")
#     print()


# %%
dataset = TextDataset([d for t in data for d in t])
print(f"Dataset size: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size = args.bs, shuffle=False)

# %%
extraction_locs = [7]
extraction_tokens = [-1] #, -2, -3, -4]
extraction_layers = list(range(model.config.num_hidden_layers))

# %%
next(iter(dataloader))

# %%


if 'assistant' in args.target_tokens:
    extraction_tokens = 'assistant'
elif 'all' in args.target_tokens:
    extraction_tokens = 'all'
else:
    extraction_tokens = [-1]


all_hidden_states = extract_hidden_states(dataloader, tokenizer, model, assistant_tag, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens, do_final_cat=True, avg_token_dim=True)


# %%
torch.cuda.empty_cache()


args.dataset_source_with_count = args.dataset_source
if args.max_samples != MAX_SAMPLES_DEFAULT:
    args.dataset_source_with_count += f'_{args.max_samples}'

if args.seed != 42:
    args.dataset_source_with_count += f'_seed{args.seed}'

root = f'{args.save_path}/{model_short_name}/{args.dataset_source_with_count}'
os.makedirs(root, exist_ok=True)

# %%
hidden_states = []
idx = 0
for task in data:
    hidden_states.append(all_hidden_states[idx:idx+len(task)])
    idx += len(task)

assert idx == len(all_hidden_states), f"Expected {len(all_hidden_states)}, got {idx}"

# %%
if 'probe' in possible_vec_types:
    all_res = {}
    for t, task_hs in enumerate(tqdm(hidden_states)):
        all_res[t] = {}
        for i, loc in enumerate(extraction_locs):
            all_res[t][i] = {}
            for l, layer in enumerate(tqdm(extraction_layers)):
                all_res[t][i][l] = {}
                for lbl in range(len(labels[t][0])):
                    print(f"Processing Task {t}, Layer {l}, Loc {loc}, col {lbl}")
                    # print(task_hs[:, l, i, -1, :].shape, labels[t][:,lbl].shape)
                    print(task_hs[:, l, i, -1, :].shape, labels[t][:,lbl].shape)
                    res = probe_classification(task_hs[:, l, i, -1, :], labels[t][:,lbl], handle_imbalanced=False, fit_intercept=False, normalize_X = False) # -1 is to get the last token
                    print("shape:", res['weights'].shape)
                    print(res['accuracy_train'], res['accuracy_test'])
                    all_res[t][i][l][lbl] = res
                    

# %%
dataset_type

# %%
# if multitask, each task is for a different concept
import matplotlib.pyplot as plt
if 'probe' in possible_vec_types:
    for i, loc in enumerate(extraction_locs):
        for c in range(len(concepts)): # concepts
            for l, layer in enumerate(extraction_layers):
                if dataset_type == 'classification-multiclass':
                    res = all_res[0][i][l][0]
                    concept_vector = torch.from_numpy(res['weights'])[c, :]                
                elif dataset_type == 'classification-multilabel':
                    res = all_res[0][i][l][c]
                    concept_vector = torch.from_numpy(res['weights'])                    
                elif dataset_type == 'multitask':
                    res = all_res[c][i][l][0]
                    concept_vector = torch.from_numpy(res['weights']).squeeze()
                else:
                    assert False
                token_str = '' if args.target_tokens == 'last' else args.target_tokens
                path = root + f'/{concepts[c]}/probe{token_str}/'
                os.makedirs(path, exist_ok=True)
                torch.save(concept_vector, path + f'layer_{layer}_loc_{loc}.pt')
            
            
            
                
    for i, loc in enumerate(extraction_locs):
        
        concept_iterator = len(hidden_states) if dataset_type != 'multitask' else len(concepts)

        for c in range(concept_iterator): # concepts
            plot_train = []
            plot_test = []
            for l, layer in enumerate(extraction_layers):
                if dataset_type == 'classification-multiclass':
                    assert c == 0
                    res = all_res[0][i][l][0]
                    plot_train.append(res['accuracy_train'].item())
                    plot_test.append(res['accuracy_test'].item())
                    
                elif dataset_type == 'classification-multilabel':
                    res = all_res[0][i][l][c]
                    plot_train.append(res['accuracy_train'].item()) # change to precision recall f1, later
                    plot_test.append(res['accuracy_test'].item())
                    
                elif dataset_type == 'multitask':
                    res = all_res[c][i][l][0]
                    plot_train.append(res['accuracy_train'].item())
                    plot_test.append(res['accuracy_test'].item())
                    
                else:
                    assert False
                
                
            token_str = '' if args.target_tokens == 'last' else args.target_tokens
            path = root + f'/{concepts[c]}/probe{token_str}/'                
            
            fig = plt.figure()
            plt.plot(plot_train, label='train')
            plt.plot(plot_test, label='test')
            plt.title(f'{model_short_name} {args.dataset_source} target {args.target_tokens} Loc {loc} \n Concept: {concepts[c]} \n max samples: {args.max_samples}')
            plt.xlabel('Layer')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(path + f'loc_{loc}.png', bbox_inches='tight', dpi=300)
            plt.show()
            
            print(f"Saved {path}")

# %%
all_vecs = {}
for t, task_hs in enumerate(tqdm(hidden_states)):
    all_vecs[t] = {}
    for i, loc in enumerate(extraction_locs):
        all_vecs[t][i] = {}
        for l, layer in enumerate(tqdm(extraction_layers)):
            all_vecs[t][i][l] = {}
            for lbl in range(len(labels[t][0])):
                print(f"Processing Task {t}, Layer {l}, Loc {loc}, column {lbl}")
                all_vecs[t][i][l][lbl] = {}
                
                for ll in labels_set[t][lbl]:
                    mask = labels[t][:, lbl] == ll
                    all_vecs[t][i][l][lbl][ll] = task_hs[mask, l, i, -1]


# %%
if 'meandiff' in possible_vec_types:
    for c, concept in enumerate(concepts):
        for i, loc in enumerate(extraction_locs):
            for l, layer in enumerate(extraction_layers):
                if dataset_type == 'multitask':
                    all_hs = deepcopy(all_vecs[c][i][l][0])
                    labels_set_ = deepcopy(labels_set[t][0])
                elif dataset_type == 'classification-multilabel':
                    all_hs = deepcopy(all_vecs[0][i][l][c])
                    labels_set_ = deepcopy(labels_set[0][t])
                elif dataset_type == 'classification-multiclass':
                    all_hs = deepcopy(all_vecs[0][i][l][0])
                    labels_set_ = deepcopy(labels_set[0][0])
                else:
                    assert False, 'not implemented'

                token_str = '' if args.target_tokens == 'last' else args.target_tokens
                path = root + f'/{concepts[c]}/meandiff{token_str}/'
                os.makedirs(path, exist_ok=True)

                pos_hs = all_hs[1]
                neg_hs = all_hs[0]                    
                concept_vector = pos_hs.mean(dim=0) - neg_hs.mean(dim=0)
                torch.save(concept_vector, path + f'layer_{layer}_loc_{loc}.pt')

# %%
import matplotlib
def get_discrete_colors(n_colors, colormap='Spectral'):
    colors = matplotlib.cm.get_cmap(colormap, n_colors)
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors

concept_iterator = len(hidden_states) if dataset_type != 'multitask' else len(concepts)

for t in range(concept_iterator):
    for i, loc in enumerate(extraction_locs):
        for l, layer in enumerate(extraction_layers):
            
            if dataset_type == 'multitask':
                all_hs = deepcopy(all_vecs[t][i][l][0])
                class_labels = [f'no_{concepts[t]}', f'{concepts[t]}']
                labels_set_ = deepcopy(labels_set[t][0])
                path = root + f'/{concepts[t]}/'
            elif dataset_type == 'classification-multilabel':
                assert t == 0
                all_hs = deepcopy(all_vecs[0][i][l][t])
                class_labels = [f'no_{concepts[t]}', f'{concepts[t]}']
                labels_set_ = deepcopy(labels_set[0][t])
                path = root + f'/{concepts[t]}/'
            elif dataset_type == 'classification-multiclass':
                assert t == 0
                all_hs = deepcopy(all_vecs[0][i][l][0])
                class_labels = deepcopy(concepts)
                labels_set_ = deepcopy(labels_set[0][0])
                path = root
            else:
                assert False, 'not implemented'
                
            
            os.makedirs(path, exist_ok=True)                



            diff_vector = all_hs[1].mean(0) - all_hs[0].mean(0)
            print(diff_vector.norm())
            labels_set_.append(len(labels_set_))
            print('appended:', labels_set_[-1])
            all_hs[labels_set_[-1]] = all_hs[0] + diff_vector[None]
            class_labels.append(f'{class_labels[0]}->{class_labels[1]}')

            all_vectors = torch.cat([all_hs[ll] for ll in labels_set_], dim=0)
            if True:
                tsne = TSNE(n_components=2, random_state=args.seed)
            else:
                tsne =  PCA(n_components=2, random_state=args.seed)
            all_vectors = tsne.fit_transform(all_vectors.cpu().numpy())
            plt.figure(figsize=(8, 8))

            lens = [len(all_hs[ll]) for ll in labels_set_]
            print("lens:", lens,)
            
            counter = 0

            colors = get_discrete_colors(len(labels_set_), colormap='tab20')[::-1]                
            for ll, label in enumerate(labels_set_):
                shape = 'o' #if ll <= 1 else '*'
                plt.scatter(all_vectors[counter:counter+lens[ll], 0], all_vectors[counter:counter+lens[ll], 1], label=f'{class_labels[ll]}', alpha=.8, marker= shape, color=colors[ll], s=30)
                counter += lens[ll]


            plt.title(f'{model_short_name} {args.dataset_source} target tokens {args.target_tokens} \n Layer {layer} Loc {loc} \n max samples: {args.max_samples}')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(path + f'layer_{layer}_loc_{loc}_{args.target_tokens}_tsne.png', bbox_inches='tight', dpi=300)
            plt.show()
        # break  

# %%



