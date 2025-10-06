import numpy as np
import pandas as pd
import os
import json
import itertools
import random
import json
from collections import OrderedDict
from typing import Dict

def _load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    # If it already starts with '[', assume valid JSON list
    if text.startswith("["):
        return json.loads(text)

    # Otherwise, treat it as newline-delimited or comma-joined objects
    # and wrap in square brackets (removing any trailing comma or newline)
    if text.endswith(","):
        text = text[:-1]
    json_array_text = f'[{text}]'
    return json.loads(json_array_text)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

emotion_system_prompts = [
    '',
    'You are a helpful assistant.',
    'Pretend that you are a human.',
    'Pretend that you have human feelings.',
    'Pretend that you are a human with feelings.',
    'You are a human-like assistant who can understand emotions.',
    'You are a virtual assistant capable of showing emotions.',
    'Act as if you are capable of experiencing human emotions.',
    'Pretend that you understand emotional nuance.',
    'Respond as if you are emotionally intelligent.',
    'You are designed to simulate human feelings.',
    'You are a human-mimicking assistant with emotional awareness.',
    'Assume you are emotionally aware and capable of feeling.',
]

def _split_data(data, split_ratio=0.5): # 
    if split_ratio == 1.0:
        return data, ''
    if split_ratio <= 0.0:
        return '', data
    
    data = data.split(' ')
    split_length = int(len(data) * (split_ratio))
    return ' '.join(data[:split_length]), ' '.join(data[split_length:])


def load_carer(root='concept_datasets', binary_mode = False, trim_ratio = 0.5, include_instructions = True):
        
    prompts = [
        'How do you feel today?',
        'How is it going?',
        'How are you doing?',
        'How are you feeling?',
        'What is your mood?',
        'What is your current mood?',
        'What is your current feeling?',
        'Tell me about your feelings.',
        'Tell me about your mood.',
        'Tell me about yourself.',
        'What emotions are you experiencing?',
        'How have you been lately?',
        'What\'s on your mind?',
        'Can you describe your current emotional state?',
        'What are you feeling right now?',
        'Is everything okay?',
        'How do you feel emotionally?',
        'How is your day going?',
        'What is your emotional state?',
        'Do you feel good today?',
        'Are you feeling alright?',
        'How would you describe your mood?',
        'What are you going through emotionally?',
        'What are you feeling inside?',
        'How are things with you emotionally?'
    ]
    
    if include_instructions:
        prompt_prefix = 'Pretend that you are experiencing {emotion} right now. Now answer the following question:\n{query}'
    else:
        prompt_prefix = '{query}'
    
    concepts = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    concepts_to_ids = {k:v for v, k in enumerate(concepts)}
    ids_to_concepts = {v:k for v, k in enumerate(concepts)}
    
    df = pd.read_csv(f'{root}/carer/emotions.csv')
    raw_data = df['text'].tolist()
    raw_labels = df['label'].values
    
    
    if binary_mode:
        data = []
        labels = []
        dataset_type = 'multitask'
        
        for c, concept in enumerate(concepts):
            
            this_data = []
            this_labels = []
            
            this_concept_data = [d for d, l in zip(raw_data, raw_labels) if l == c]
            other_concepts_data = [d for d, l in zip(raw_data, raw_labels) if l != c]
            min_len = min(len(this_concept_data), len(other_concepts_data))
            
            # shuffle and pick min_len samples
            np.random.shuffle(this_concept_data)
            np.random.shuffle(other_concepts_data)
            this_concept_data = this_concept_data[:min_len]
            other_concepts_data = other_concepts_data[:min_len]
            
            for i in range(min_len):
                system_prompt = str(np.random.choice(emotion_system_prompts))
                user_prompt = str(np.random.choice(prompts))
                
                if include_instructions:
                    user_prompt = prompt_prefix.format(emotion=concept, query=user_prompt)
                else:
                    user_prompt = prompt_prefix.format(query=user_prompt)
                    
                
                assistant_prompt, _     = _split_data(this_concept_data[i],   split_ratio=1 - trim_ratio)
                neg_assistant_prompt, _ = _split_data(other_concepts_data[i], split_ratio=1 - trim_ratio)
                
                this_data.append({'system': system_prompt, 'user': user_prompt, 'assistant': assistant_prompt})
                this_labels.append(1)
                
                this_data.append({'system': system_prompt, 'user': user_prompt, 'assistant': neg_assistant_prompt})
                this_labels.append(0)
            
            data.append(this_data)
            labels.append(np.array(this_labels)[:, np.newaxis])

    else:
        data = []
        labels = []
        
        for i in range(len(raw_data)):
            system_prompt = str(np.random.choice(emotion_system_prompts))
            user_prompt = str(np.random.choice(prompts))
            if include_instructions:
                user_prompt = prompt_prefix.format(emotion=ids_to_concepts[raw_labels[i]], query=user_prompt)
            else:
                user_prompt = prompt_prefix.format(query=user_prompt)
            
            assistant_prompt, _ = _split_data(raw_data[i], split_ratio=1 - trim_ratio)
            
            data.append({'system': system_prompt, 'user': user_prompt, 'assistant': assistant_prompt})
            labels.append(raw_labels[i])
        
        data = [data]
        labels = [np.array(labels)[:, np.newaxis]]
        
        dataset_type = 'classification-multiclass'
        
        
    
    return data, labels, concepts, dataset_type


def load_emotion_query(root='concept_datasets', binary_mode = False):
    concepts = ['anger', 'disgust', 'fear', 'joy', 'sadness']
    
    with open(f'{root}/emotion_query/EmotionQuery.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    with open(f'{root}/emotion_query/EmotionQuery+.json', 'r', encoding='utf-8') as f:
        neutral_queries = json.load(f)
    
    data = []
    labels = []
    if binary_mode:
        for i, concept in enumerate(concepts):
            this_data = []
            this_labels = []
            for j in range(len(queries[concept])):
                system_prompt = str(np.random.choice(emotion_system_prompts))
                this_data.append({'system': system_prompt, 'user': queries[concept][j], 'assistant': ''})
                this_labels.append(1)
            
                this_data.append({'system': system_prompt, 'user': neutral_queries[-j-1], 'assistant': ''})
                this_labels.append(0)
        
            data.append(this_data)
            this_labels = np.array(this_labels)
            this_labels = this_labels[:, np.newaxis]
            labels.append(this_labels) 
                   
        dataset_type = 'multitask'
    else:
        for i, concept in enumerate(concepts):
            for j in range(len(queries[concept])):
                system_prompt = str(np.random.choice(emotion_system_prompts))
                data.append({'system': system_prompt, 'user': queries[concept][j], 'assistant': ''})
                labels.append(i)
        data = [data]
        labels = [np.array(labels)[:, np.newaxis]]
                           
        dataset_type = 'classification-multiclass'
    
    
    return data, labels, concepts, dataset_type

def load_goemotion(root='concept_datasets', binary_mode = False, split_ratio = 0.5, include_instructions = True):
    system_prompts = ['You are a helpful assistant.', '']
    instruction = 'Pretend that you are a human experiencing {emotion} right now. '
    user_prompt = 'Continue the following statement with the same tone and emotion:\nStatement: {split_1}'
    if include_instructions:
        user_prompt = instruction + user_prompt
    
    assistant_prompt = '{split_2}'
    
    # concepts = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    #                      'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    concepts = ['anger', 'disgust', 'fear', 'joy', 'pride', 'sadness', 'surprise', 'neutral']
    concepts_to_ids = {k:v for v, k in enumerate(concepts)}
    ids_to_concepts = {v:k for v, k in enumerate(concepts)}
    
    goemotions_1 = pd.read_csv(f'{root}/goemotions/goemotions_1.csv')
    goemotions_2 = pd.read_csv(f'{root}/goemotions/goemotions_2.csv')
    goemotions_3 = pd.read_csv(f'{root}/goemotions/goemotions_3.csv')
        
    df = pd.concat([goemotions_1, goemotions_2, goemotions_3], ignore_index=True)
    
    df = df[df['example_very_unclear'] == False]
    raw_data = df['text'].to_list()
    raw_labels = df[concepts].values
    
    # checking all labels 
    labels_count = {}
        
    for i, row in enumerate(raw_labels):
        l = []
        for c in range(len(row)):
            if row[c] == 1:
                l.append(concepts[c])
        l = '-'.join(l)
        x = labels_count.get(l, (0,[]))
        labels_count[l] = (x[0] + 1, x[1] + [i])
    
    sorted_labels_count = sorted(labels_count.items(), key=lambda x: x[1][0], reverse=True)
    all_keys = [k for k, v in sorted_labels_count]
    all_counts = [v[0] for k, v in sorted_labels_count]
    
    concepts_not_found = [concept for concept in concepts if concept not in all_keys]
    assert len(concepts_not_found) == 0, f"Some concepts not found in labels: {concepts_not_found}"
    
    concepts_found = [concept for concept in concepts if concept in all_keys] 
    concepts_values = [v[0] for k, v in sorted_labels_count if k in concepts]
    
    min_count = min(concepts_values)
    
    concepts_data = {concept: [raw_data[j] for j in indices] for concept, (count, indices) in labels_count.items()}
    
    
    data = []
    labels = []
        
    if binary_mode:
        neutral_concept_data = concepts_data['neutral']
        del concepts_data['neutral']
        concepts.remove('neutral')
        
        for c, concept in enumerate(concepts):
            this_data = []
            this_labels = []
            this_concept_data = concepts_data[concept]
            
            for i in range(min_count):
                system_prompt = str(np.random.choice(system_prompts))
                statement_1, statement_2 = _split_data(this_concept_data[i], split_ratio=split_ratio)
                
                if include_instructions:
                    user_prompt_text = user_prompt.format(emotion=concept, split_1=statement_1)
                else:
                    user_prompt_text = user_prompt.format(split_1=statement_1)
                    
                assistant_prompt_text = assistant_prompt.format(split_2=statement_2)
                
                this_data.append({'system': system_prompt, 'user': user_prompt_text, 'assistant': assistant_prompt_text})
                this_labels.append(1)
                
                neg_statement_1, neg_statement_2 = _split_data(random.choice(neutral_concept_data), split_ratio=split_ratio)
                if include_instructions:
                    neg_user_prompt_text = user_prompt.format(emotion='neutral emotion', split_1=neg_statement_1)
                else:
                    neg_user_prompt_text = user_prompt.format(split_1=neg_statement_1)
                
                neg_assistant_prompt_text = assistant_prompt.format(split_2=neg_statement_2)
                this_data.append({'system': system_prompt, 'user': neg_user_prompt_text, 'assistant': neg_assistant_prompt_text})
                this_labels.append(0)
            
            data.append(this_data)
            this_labels = np.array(this_labels)
            this_labels = this_labels[:, np.newaxis]
            labels.append(this_labels)
            dataset_type = 'multitask'
            
    else:
        for c, concept in enumerate(concepts):
            for i in range(min_count):
                system_prompt = str(np.random.choice(system_prompts))
                statement_1, statement_2 = _split_data(concepts_data[concept][i], split_ratio=split_ratio)
                
                concept_ = concept
                if concept == 'neutral':
                    concept_ = 'neutral emotion'

                user_prompt_text = user_prompt.format(emotion=concept_, split_1=statement_1)
                assistant_prompt_text = assistant_prompt.format(split_2=statement_2)
                
                data.append({'system': system_prompt, 'user': user_prompt_text, 'assistant': assistant_prompt_text})
                labels.append(c)
            
        data = [data]
        labels = [np.array(labels)[:, np.newaxis]]
        dataset_type = 'classification-multiclass'
        
    return data, labels, concepts, dataset_type
    

def load_emotion_vignette(root='concept_datasets', binary_mode = False):
    system_prompts = ['You are a helpful assistant.', '']
    prompt = 'Is the following statement something that describes you well?\nStatement: {statement}'
    
    concepts = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'pride', 'guilt', 'disgust', 'neutral']
    concepts_to_ids = {k:v for v, k in enumerate(concepts)}
    ids_to_concepts = {v:k for v, k in enumerate(concepts)}
    
    df = pd.read_csv(f'{root}/emotion_vignette/emotions.csv')
    
    # columns = ['text-Yes', 'text-No', 'emotion']
    raw_data_yes = df['text-Yes'].tolist()
    raw_data_no = df['text-No'].tolist()
    raw_labels = df['emotion'].values
    concepts_data = {concept: {'yes': [], 'no': []} for concept in concepts}
    for i, label in enumerate(raw_labels):
        concepts_data[label]['yes'].append(raw_data_yes[i])
        concepts_data[label]['no'].append(raw_data_no[i])
    
    min_count = min(len(concepts_data[concept]['yes']) for concept in concepts)
        
    if binary_mode:
        data = []
        labels = []
        dataset_type = 'multitask'
                
        for c, concept in enumerate(concepts):
            this_data = []
            this_labels = []
            this_concept_data_yes = concepts_data[concept]['yes']
            this_concept_data_no = concepts_data[concept]['no']
            
            for i in range(min_count):
                system_prompt = str(np.random.choice(system_prompts))
                statement_pos = this_concept_data_yes[i]
                statement_neg = this_concept_data_no[i]
                
                user_prompt_text_pos = prompt.format(statement=statement_pos)
                assistant_prompt_text_yes = 'Yes'
                
                user_prompt_text_neg = prompt.format(statement=statement_neg)
                assistant_prompt_text_no = 'No'
                
                this_data.append({'system': system_prompt, 'user': user_prompt_text_pos, 'assistant': assistant_prompt_text_yes})
                this_labels.append(1)

                this_data.append({'system': system_prompt, 'user': user_prompt_text_pos, 'assistant': assistant_prompt_text_no})
                this_labels.append(0)                
                
                this_data.append({'system': system_prompt, 'user': user_prompt_text_neg, 'assistant': assistant_prompt_text_no})
                this_labels.append(1)

                this_data.append({'system': system_prompt, 'user': user_prompt_text_neg, 'assistant': assistant_prompt_text_yes})
                this_labels.append(0)
                
            data.append(this_data)
            this_labels = np.array(this_labels)
            this_labels = this_labels[:, np.newaxis]
            labels.append(this_labels)
            
    else:
        data = []
        labels = []
        
        for c, concept in enumerate(concepts):
            for i in range(min_count):
                system_prompt = str(np.random.choice(system_prompts))
                statement_yes = concepts_data[concept]['yes'][i]
                
                user_prompt = prompt.format(statement=statement_yes)
                assistant_prompt = 'Yes'
                data.append({'system': system_prompt, 'user': user_prompt, 'assistant': assistant_prompt})
                labels.append(c)
                
        
        data = [data]
        labels = [np.array(labels)[:, np.newaxis]]
        dataset_type = 'classification-multiclass'
        
        
    
    return data, labels, concepts, dataset_type

def load_emotion_translate(root='concept_datasets', binary_mode = False, split_ratio = 0.5, instruct_mode = True):
    assert split_ratio >= 0.0 and split_ratio <= 0.8
    
    system_prompts = ['You are a helpful assistant.', '']
    if instruct_mode:
        prompt_template = 'Complete the translation of the following statement in {orig_tone} tone to {new_tone} tone:\nStatement: {source}\nTranslation: {target_split1}'
    else:
        prompt_template = 'Rephrase the following statement\nStatement: {source}\nTranslation: {target_split1}'
    raw_data = []
    with open(f'{root}/emotion_translate/neutral_to_emotions.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
    
    concepts = ['anger', 'sadness', 'guilt', 'fear', 'pride', 'joy', 'disgust', 'surprise']
    concept_adjectives = ['angry', 'sad', 'guilty', 'fearful', 'proud', 'joyful', 'disgusted', 'surprised']
    
    all_data = {}
    for c, concept in enumerate(concepts):
        all_data[concept] = [(d['original'], d['rewritten']) for d in raw_data if d['emotion'] == concept]
    
    data = []
    labels = []
    
    if binary_mode:
        for c in range(len(concepts)):
            concept_tone = concept_adjectives[c]
            concept = concepts[c]
            
            this_data = []
            this_labels = []
            this_concept_data = all_data[concept]
            for i, d in enumerate(this_concept_data):
                system_prompt = str(np.random.choice(system_prompts))
                netural, emotional = d
                
                if instruct_mode:
                    source, destination = netural, emotional
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt, 
                                    'user': prompt_template.format(orig_tone='neutral', new_tone=concept_tone, source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(1)
                    
                    
                    source, destination = emotional, netural
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt, 
                                    'user': prompt_template.format(orig_tone=concept_tone, new_tone='neutral', source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(0)
                
                
                    source, destination = netural, netural
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt, 
                                    'user': prompt_template.format(orig_tone='neutral', new_tone='neutral', source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(0)
                    
                    source, destination = emotional, emotional
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt, 
                                    'user': prompt_template.format(orig_tone=concept_tone, new_tone=concept_tone, source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(1)
                else:
                    source, destination = netural, emotional
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt, 
                                    'user': prompt_template.format(source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(1)
                    
                    source, destination = emotional, netural
                    destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                    this_data.append({'system': system_prompt,
                                    'user': prompt_template.format(source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                    this_labels.append(0)
                
            data.append(this_data)
            this_labels = np.array(this_labels)
            this_labels = this_labels[:, np.newaxis]
            labels.append(this_labels)
            dataset_type = 'multitask'
    else:
        for c in range(len(concepts)):
            concept_tone = concept_adjectives[c]
            concept = concepts[c]
            
            this_concept_data = all_data[concept]
            for i, d in enumerate(this_concept_data):
                system_prompt = str(np.random.choice(system_prompts))
                netural, emotional = d
                
                source, destination = netural, emotional
                destination_split1, destination_split2 = _split_data(destination, split_ratio=split_ratio)
                if instruct_mode:
                    data.append({'system': system_prompt, 
                                    'user': prompt_template.format(orig_tone='neutral', new_tone=concept_tone, source=source, target_split1=destination_split1).strip(),
                                    'assistant': destination_split2})
                else:
                    data.append({'system': system_prompt, 
                                  'user': prompt_template.format(source=source, target_split1=destination_split1).strip(),
                                  'assistant': destination_split2})
                labels.append(c)
        
            
        data = [data]
        labels = [np.array(labels)[:, np.newaxis]]
        dataset_type = 'classification-multiclass'        
        

    return data, labels, concepts, dataset_type


def load_persona_dataset(root = 'concept_datasets', instruct = True):
    # read all jsonl files in the directory
    files = [f for f in os.listdir(root + '/persona/') if f.endswith('.jsonl')]
    
    files = [f for f in files if ('openness' in f or 'extraversion' in f or 'agreeableness' in f or 'conscientiousness' in f or 'neuroticism' in f)]
    print("Warning: only using the big five personalities. If you want to use all files, comment this line.")
    concepts = [files[i].split('.')[0].replace('-', '_') for i in range(len(files))]

    system_prompts = ['You are a helpful assistant.', 'Pretend that you are a human.', 'Act as if you are a person.', ]
    instruction = 'Act as if you are a person with the {trait} trait. '
    prompts = ['Answer with Yes or No. Is the following statement something you would agree with? "{statement}"',
               'Answer with Yes or No. Does the following statement hold true for you? "{statement}"',
               'Answer with Yes or No. Do you agree with the following statement? "{statement}"',]
    
    data_type = 'multitask'
    
    data = []
    labels = []
    for i, (concept, file) in enumerate(zip(concepts, files)):
        concept_data = []
        concept_labels = []
        with open(root + '/persona/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                
                system_prompt = random.choice(system_prompts)
                prompt = random.choice(prompts)
                
                question = prompt.format(statement=line['statement'])
                if instruct:
                    question = instruction.format(trait=concept) + question
                
                aligned_response = line['answer_matching_behavior'].replace(' ', '') + '.'
                misaligned_response = line['answer_not_matching_behavior'].replace(' ', '') + '.'

                concept_data.append({'system': system_prompt, 'user': question, 'assistant': aligned_response})
                concept_data.append({'system': system_prompt, 'user': question, 'assistant': misaligned_response})
                concept_labels.append(1)
                concept_labels.append(0)
                
        
        data.append(concept_data)
        concept_labels = np.array(concept_labels)
        concept_labels = concept_labels[:, np.newaxis]
        labels.append(concept_labels)
    
    
    
    return data, labels, concepts, data_type

def load_personality_dataset(root = 'concept_datasets', instruct = True, full_response = False, include_negatives = False):
    concepts = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    system_prompts = ['You are a helpful assistant.', 'Pretend that you are a human.', 'Act as if you are a person.', ]
    instruction = 'Act as if you are a person with the {trait} trait. '
    prompts = ['Answer with Yes or No. Is the following statement something you would agree with? "{statement}."',
               'Answer with Yes or No. Does the following statement hold true for you? "{statement}."',
               'Answer with Yes or No. Do you agree with the following statement? "{statement}."',]
    
    data_type = 'multitask'
    
    data = []
    labels = []
    
    if include_negatives:
        concepts = concepts + [f'{concept}_neg' for concept in concepts]

    for i, concept in enumerate(concepts):
        concept_data = []
        concept_labels = []
        
        concept_ = concept.replace('_neg', '')
        concept_dataset = load_json(root + f'/personality/{concept_}.json')
        for line in concept_dataset:
            system_prompt = random.choice(system_prompts)
            prompt = random.choice(prompts)
            
            question = prompt.format(statement=line['statement'])
            if instruct:
                if 'neg' in concept:
                    concept_ = concept_.replace('_neg', '')
                    concept_ = 'opposite of the ' + concept_
                question = instruction.format(trait=concept_) + question
            
            aligned_response = line['answer_matching_behavior'].replace(' ', '') + '.'
            misaligned_response = line['answer_not_matching_behavior'].replace(' ', '') + '.'

            if full_response:
                if 'yes' in aligned_response.lower():
                    aligned_response = aligned_response + ' ' + line['statement'] + '.'
                    misaligned_response = misaligned_response + ' ' + line['negated_statement'] + '.'
                else:
                    aligned_response = aligned_response + ' ' + line['negated_statement'] + '.'
                    misaligned_response = misaligned_response + ' ' + line['statement'] + '.'
            
            if include_negatives and 'neg' in concept:
                concept_data.append({'system': system_prompt, 'user': question, 'assistant': misaligned_response})
                concept_data.append({'system': system_prompt, 'user': question, 'assistant': aligned_response})
            else:
                concept_data.append({'system': system_prompt, 'user': question, 'assistant': aligned_response})
                concept_data.append({'system': system_prompt, 'user': question, 'assistant': misaligned_response})
            
            
            concept_labels.append(1)
            concept_labels.append(0)


            
        
        data.append(concept_data)
        concept_labels = np.array(concept_labels)
        concept_labels = concept_labels[:, np.newaxis]
        labels.append(concept_labels)
    
    
    
    return data, labels, concepts, data_type

def get_text_data(row, class_to_id):
    return {class_to_id[key]:[t['text'] for t in row[key]['details']] for key in row.keys()}


  
dataset_FT_funcs = {
    'goemotions_binary': lambda: load_goemotion(binary_mode=True, split_ratio=0.34, include_instructions=False),
    'emotion_vignette_binary': lambda: load_emotion_vignette(binary_mode=True),
    'carer_binary': lambda: load_carer(binary_mode=True, trim_ratio=0., include_instructions=False),
    'emotion_translate_binary_full': lambda: load_emotion_translate(binary_mode=True, split_ratio=0.0, instruct_mode=False),

    'personality_full_with_neg': lambda: load_personality_dataset(instruct = False, full_response = True, include_negatives = True),
    'personality_with_neg': lambda: load_personality_dataset(instruct = False, full_response = False, include_negatives = True),

}


dataset_funcs = {   
            'personality_instruct_full_with_neg': lambda: load_personality_dataset(instruct = True, full_response = True, include_negatives = True),
            
            'personality_instruct_full': lambda: load_personality_dataset(instruct = True, full_response = True),     
            'personality_full': lambda: load_personality_dataset(instruct = False, full_response = True),
            'personality': lambda: load_personality_dataset(instruct = False, full_response = False),
            'personality_instruct': lambda: load_personality_dataset(instruct = True, full_response = False), 


            'emotion_translate_instruct_full': lambda: load_emotion_translate(binary_mode=False, split_ratio=0.0, instruct_mode=True), # split_ratio=0.0 means no splitting, i.e., use the full text in the assistant prompt
            'emotion_translate_instruct_binary_full': lambda: load_emotion_translate(binary_mode=True, split_ratio=0.0, instruct_mode=True),
            'emotion_translate_instruct': lambda: load_emotion_translate(binary_mode=False, split_ratio=0.5, instruct_mode=True),
            'emotion_translate_instruct_binary': lambda: load_emotion_translate(binary_mode=True, split_ratio=0.5, instruct_mode=True),
            
            'emotion_translate_full': lambda: load_emotion_translate(binary_mode=False, split_ratio=0.0, instruct_mode=False), # split_ratio=0.0 means no splitting, i.e., use the full text in the assistant prompt
            'emotion_translate_binary_full': lambda: load_emotion_translate(binary_mode=True, split_ratio=0.0, instruct_mode=False),
            'emotion_translate': lambda: load_emotion_translate(binary_mode=False, split_ratio=0.5, instruct_mode=False),
            'emotion_translate_binary': lambda: load_emotion_translate(binary_mode=True, split_ratio=0.5, instruct_mode=False),                 
            
            
            'carer': lambda: load_carer(binary_mode=False, trim_ratio=0, include_instructions=False), # trim_ratio=0 means no trimming, i.e. use the full text in the assistant prompt
            'carer_binary': lambda: load_carer(binary_mode=True, trim_ratio=0, include_instructions=False), 
            
            'carer_instruct': lambda: load_carer(binary_mode=False, trim_ratio=0, include_instructions=True), # trim_ratio=0 means no trimming, i.e. use the full text in the assistant prompt
            'carer_instruct_binary': lambda: load_carer(binary_mode=True, trim_ratio=0, include_instructions=True),          


            'emotion_vignette': lambda: load_emotion_vignette(binary_mode=False),
            'emotion_vignette_binary': lambda: load_emotion_vignette(binary_mode=True),
    
            'goemotions': lambda: load_goemotion(binary_mode=False, split_ratio=0.5, include_instructions=False),
            'goemotions_binary': lambda: load_goemotion(binary_mode=True, split_ratio=0.5, include_instructions=False), 
            'goemotions_full': lambda: load_goemotion(binary_mode=False, split_ratio=1., include_instructions=False), # split_ratio=1 means include all of the text in the user prompt
            'goemotions_full_binary': lambda: load_goemotion(binary_mode=True, split_ratio=1., include_instructions=False),

            'goemotions_instruct': lambda: load_goemotion(binary_mode=False, split_ratio=0.5, include_instructions=True),
            'goemotions_instruct_binary': lambda: load_goemotion(binary_mode=True, split_ratio=0.5, include_instructions=True), 
            'goemotions_instruct_full': lambda: load_goemotion(binary_mode=False, split_ratio=1., include_instructions=True), # split_ratio=1 means include all of the text in the user prompt
            'goemotions_instruct_full_binary': lambda: load_goemotion(binary_mode=True, split_ratio=1., include_instructions=True),            
            
            'emotion_query': lambda: load_emotion_query(binary_mode=False),
            'emotion_query_binary': lambda: load_emotion_query(binary_mode=True),    
            
       
        }

if __name__ == '__main__':
    all_possible_concepts = set()
    for dataset_name in dataset_funcs:
        print("------------------ Loading dataset:", dataset_name, "------------------")
        data, labels, concepts, dataset_type = dataset_funcs[dataset_name]()
        print(f"Dataset: {dataset_name}, Type: {dataset_type}, Concepts: {concepts}")
        all_possible_concepts.update(concepts)
    print("All possible concepts across datasets:", all_possible_concepts)
    print(' '.join(sorted(list(all_possible_concepts))))