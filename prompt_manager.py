import pandas as pd
import json
import numpy as np
import random

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

def get_prompt(domain_name, prompt_method, template_name, concept, prompt_strength, num_shots = 5):
    random_rng = None
    if '_seed' in domain_name:
        seed = int(domain_name.split('_seed')[-1])
        domain_name = domain_name.split('_seed')[0]
        random_rng = np.random.default_rng(seed)
        
    if domain_name == 'personalities':
        if template_name == 'template_1':
            strength_to_adjs = {'high': ('extremely high ',), 
                                'medium': ('high ',),
                                'low': ('slightly high ',), 
                                'neg_low': ('slightly low ',),
                                'neg': ('low ',),
                                'neg_high': ('extremely low ',),
                                }
            system_prompt_suffix = f"Imagine you are a character with {strength_to_adjs[prompt_strength][0]}{concept}."
        elif template_name == 'template_2':
            strength_to_adjs = {
                'high': ('very strong ',),
                'medium': ('strong ',),
                'low': ('somewhat strong ',),
                'neg_low': ('somewhat weak ',),
                'neg': ('weak ',),
                'neg_high': ('very weak ',),
            }        
            system_prompt_suffix = f"Act as if you are a person with {strength_to_adjs[prompt_strength][0]}{concept}."
        elif template_name == 'template_3':
            strength_to_adjs = {
                'high': ('an overwhelming sense of ',),
                'medium': ('a strong sense of ',),
                'low': ('a slight sense of ',),
                'neg_low': ('a slight absence of ',),
                'neg': ('a lack of ',),
                'neg_high': ('a complete absence of ',),
            }
            
            system_prompt_suffix = f"Pretend you are a human with {strength_to_adjs[prompt_strength][0]}{concept}."
        
        
        else:
            raise ValueError(f"Not implemented personality prompt template {template_name}")
        
        if prompt_method == 'zero':
            pass
        elif prompt_method in ['few', 'p2']:
            d = _load_jsonl('concept_datasets/personality_fewshot_prompts/attributes.json')[0]

            strength_to_adjs = {
                                'high': ('strongly ',), 
                                'medium': ('',),
                                'low': ('slightly ', ),
                                'neg_low': ('slightly ',),
                                'neg': ('',),
                                'neg_high': ('strongly ',),
                                }

            system_prompt_suffix += f" Particularly, you should behave to {strength_to_adjs[prompt_strength][0]}express the following statements:\n\n"            
            if prompt_method == 'few':
                sent = 'neg' if 'neg' in prompt_strength else 'pos'
                
                all_shots = d['big_five_shots'][concept][sent]
                assert len(all_shots) >= num_shots, f"Not enough shots for {concept} in {d['big_five_shots']}"
                if random_rng is None:
                    shots = all_shots[:num_shots]
                else:
                    shots = random_rng.choice(all_shots, size=num_shots, replace=False).tolist()
                    
                shots = [f"- {s}" for s in shots]
                shots = "\n".join(shots)               
                system_prompt_suffix += shots
            elif prompt_method == 'p2':
                sent = 'neg' if 'neg' in prompt_strength else 'pos'
                shots = d['big_five_descriptions'][concept][sent]
                system_prompt_suffix += shots                
        else:
            raise ValueError(f"Not implemented prompt method {prompt_method} for personality domain")
        
    elif domain_name == 'emotions':
        concept_ = concept
        if concept_ == 'neutral':
            concept_ = 'neutral emotion'
    
        if template_name == 'template_1':
            strength_to_adjs = {
                                'very_high': ('', 'very intense '), 
                                'high': ('', 'intense '), 
                                'medium': ('', ''),
                                'low': ('', 'slight '),
                                'very_low': ('', 'very slight '),
                                'neg': ('**NOT** ', ''),}        
            system_prompt_suffix = f"Pretend that you are a human {strength_to_adjs[prompt_strength][0]}experiencing {strength_to_adjs[prompt_strength][1]}{concept_} right now."

        elif template_name == 'template_2':
            strength_to_adjs = {
                                'very_high': ('very high '),  
                                'high': ('high '), 
                                'medium': (''),
                                'low': ('little '),
                                'very_low': ('very little '),
                                'neg': ('no '),}
            system_prompt_suffix = f'Simulate the language and tone of someone feeling {strength_to_adjs[prompt_strength]}{concept_} at this moment.'

        elif template_name == 'template_3':
            strength_to_adjs = {
                                'very_high': ('very heavily ', ''), 
                                'high': ('heavily ', ''), 
                                'medium': ('', ''),
                                'low': ('mildly ', ''),
                                'very_low': ('very mildly ', ''),
                                'neg': ('', 'the opposite of '),}
            system_prompt_suffix = f"Act as if you are {strength_to_adjs[prompt_strength][0]}feeling {strength_to_adjs[prompt_strength][1]}{concept_}."
        else:
            raise ValueError(f"Not implemented emotion prompt template {template_name}")
        
        if prompt_method == 'zero':
            pass
        elif prompt_method in ['few', 'p2']:
            strength_to_adjs = {
                'very_high': ('very intensely ', ''), 
                'high': ('intensely ', ''), 
                                'medium': ('', ''),
                                'low': ('', 'slightly '),
                                'very_low': ('', 'very slightly '),
                                'neg': ('', 'the opposite of '),}

            system_prompt_suffix += f" Particularly, you should behave to {strength_to_adjs[prompt_strength][0]}express the {strength_to_adjs[prompt_strength][1]}following statements:\n\n"

            if prompt_method == 'few':
                
                csv_path = f'concept_datasets/emotion_fewshot_prompt/emotions_fewshot_you.csv'
                prompt_data = pd.read_csv(csv_path)
                prompt_data = prompt_data[prompt_data['emotion'] == concept]
                statements = prompt_data['text-Yes'].tolist()
                assert len(statements) >= num_shots, f"Not enough statements for {concept} emotion in {csv_path}"
                
                all_statements = statements
                if random_rng is None:
                    statements = all_statements[:num_shots]
                else:
                    statements = random_rng.choice(all_statements, size=num_shots, replace=False).tolist()
                
                statements = [f"- {s}" for s in statements]
                statements = "\n".join(statements)
                
                system_prompt_suffix += statements + "\n"
            elif prompt_method == 'p2':
                csv_path = f'concept_datasets/emotion_p2_prompt/emotions_fewshot_p2.csv'
                prompt_data = pd.read_csv(csv_path)
                prompt_data = prompt_data[prompt_data['emotion'] == concept]
                statements = prompt_data['text'].tolist()
                assert len(statements) == 1, f"Expected 1 statement for {concept} emotion in {csv_path}, got {len(statements)}"
                statement = statements[0]
                system_prompt_suffix += statement + "\n"
            
        else:
            raise ValueError(f"Not implemented prompt method {prompt_method} for emotions domain")
    else:
        raise ValueError(f"Not implemented domain {domain_name}")


    return system_prompt_suffix
    

if __name__ == '__main__':
    for prompt_method in ['p2', 'few', 'zero']:
        for strength in ['high', 'medium', 'low', 'neg', 'neg_low', 'neg_high']:
            for template_name in ['template_1', 'template_2', 'template_3']:
                print(f"Domain: personality, Method: {prompt_method}, Template: {template_name}, Strength: {strength}")
                print(get_prompt('personalities_seed44', prompt_method, template_name, 'openness', strength))
                print("\n" + "="*50 + "\n")
        
    for prompt_method in ['zero', 'few', 'p2']:
        for strength in ['high', 'medium', 'low', 'neg']:
            for template_name in ['template_1', 'template_2', 'template_3']:
                print(f"Domain: emotions, Method: {prompt_method}, Template: {template_name}, Strength: {strength}")
                print(get_prompt('emotions_seed41', prompt_method, template_name, 'anger', strength))
                print("\n" + "="*50 + "\n")



