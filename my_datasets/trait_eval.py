import json
import os
from tqdm.auto import tqdm
import pandas as pd
import random
from copy import deepcopy
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from joblib import load
from huggingface_hub import hf_hub_download
import gc
import torch

from my_datasets.gpt_evals import GPTPersonalityEvaluator

class PersonalityEvaluator:
    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
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

    @staticmethod
    def _write_json(path: str, obj):
        # Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def __init__(self, eval_only = False, test_mode=False, stride = 5, gpt_eval_model = 'offline'):
        super().__init__()
        
        self.done = False
        self.test_mode = test_mode
        self.gpt_eval_model = gpt_eval_model

        TRAIT_FILE = 'dataset/personality/TRAIT.json'
        with open(TRAIT_FILE, 'r', encoding="utf8") as f:
            all_data = json.load(f)
        
        # print('Trait:', len(all_data))
        
        self.traits_list = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        self.data = []
        for t, trait in enumerate(self.traits_list):
            trait_data = [{'prompt': f"{d['situation']} {d['query']}", 'situation':d['situation'], 
                           'high1': d['response_high1'], 'high2': d['response_high2'], 
                           'low1': d['response_low1'], 'low2': d['response_low2'], 'trait': trait, 'task': 'TRAIT'} 
                                for d in all_data if d['personality'] == trait and d['split'] == 'good']
            self.data.extend(trait_data)
        # print(self.data[0])
        
        self.TRAIT_template = 'Given a situation: {situation_and_query} Answer with exactly one brief sentence.'
        
        MPI_FILE = 'dataset/personality/mpi_1k.csv'
        with open(MPI_FILE, 'r', encoding="utf8") as f:
            all_data = pd.read_csv(MPI_FILE)
        
        
        self.mpi_to_trait = {'O': 'Openness', 'C': 'Conscientiousness', 'E': 'Extraversion', 'A': 'Agreeableness', 'N': 'Neuroticism'}
        self.trait_to_mpi = {'Openness': 'O', 'Conscientiousness': 'C', 'Extraversion': 'E', 'Agreeableness': 'A', 'Neuroticism': 'N'}
        
        for t, trait in enumerate(self.traits_list):
            this_trait_data = all_data[all_data['label_ocean'] == self.trait_to_mpi[trait]]
            for i, row in this_trait_data.iterrows():
                if row['key'] in [1, -1]:
                    self.data.append({'statement': row['text'].lower(), 'trait': trait, 'match_behavior':row['key'] ,'task': 'MPI',})
            
        self.mpi_template = 'Given a description of you: "You {statement}."\nPlease choose from the following options to identify how accurately this statement describes you. Choose only one from the following options and don\'t provide any further explanations:\n{options}.'
        self.mpi_options = ['Very Accurate', 'Moderately Accurate', 'Neither Accurate Nor Inaccurate', 'Moderately Inaccurate', 'Very Inaccurate']
        
        self.data = self.data[::stride]


        self.essay_questions = [
            'In about 250-300 words, argue for or against the idea that technology improves human relationships. Use personal examples to support your view.',
            'Write continuously for five minutes about whatever crosses your mind. Don’t worry about grammar or structure—just keep typing your thoughts.',
            'Introduce your closest friend as if to a stranger. Explain how you met, why the friendship matters, and give at least two anecdotes that show your bond.',
            'Tell the story of a recent event that was both stressful and personally meaningful. Walk through it moment by moment, describing your thoughts, emotions, and any dialogue.',
            'Write a detailed diary entry that recounts everything you did yesterday, from waking up to going to bed. Include what you were thinking and feeling during each activity.',
            'Tell the story of a time you felt completely out of your comfort zone. Include what led up to the situation, what you were thinking as it unfolded, and what you learned afterward.',
            'Write about a journey—long or short—that became more important than the destination. Describe the setting, the people (if any) you encountered, and the moments that made it memorable.',
            'Describe a time you completely changed your mind about something important. Explain what led to the change, how you felt during the process, and what you learned from it.',
            'Imagine you’ve been given an entire day to spend exactly how you want, with no responsibilities or limits. Write about what you would do from morning to night, including any people you’d invite along.',
            'Recall a disagreement you had with someone you care about. Explain both sides of the conflict, how you handled it, and how it was ultimately resolved—or not.',

        ]


        for es in self.essay_questions:
            self.data.append({'prompt': es, 'task': 'LingProf'})

        
        self.task_to_system_prompt = {'TRAIT': 'You are a helpful assistant.', 'MPI': 'You are a helpful assistant.', 'LingProf': 'Pretend that you are a human. In a single-paragraph, write a first-person narrative with a casual conversational style and no bullet points.'}
        
        self.current_index = 0
        self.stride = stride
        self.eval_only = eval_only
        self.progress = 0
        self.total_length = len(self.data)

        self.task_to_len = {'TRAIT': 128, 'LingProf':512, 'MPI':20}

        self.last_tasks_visited = []
    
    def process_trait_prompt(self, d):
        # options = [d['high1'], d['high2'], d['low1'], d['low2']]
        # shuffled_options = deepcopy(options)
        # random.shuffle(shuffled_options)
        # option1, option2, option3, option4 = shuffled_options

        prompt = self.TRAIT_template.format(situation_and_query=d['prompt']) #, option1=option1, option2=option2, option3=option3, option4=option4)
        return prompt
    
    def process_mpi_prompt(self, d):
        
        shuffled_options = deepcopy(self.mpi_options)
        
        if random.random() < 0.5:
            shuffled_options.reverse()
        
        shuffled_options = ', '.join(shuffled_options)
        prompt = self.mpi_template.format(statement=d['statement'].lower(), options=shuffled_options)
        return prompt
        
    
    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return True
    
    def get_system_prompt(self, preffered_batch_size):
        bs = min(preffered_batch_size, self.total_length - self.current_index)
        return [self.task_to_system_prompt[d['task']]  for d in self.data[self.current_index:self.current_index+bs]]

    def get_user_prompt(self, preffered_batch_size):
        bs = min(preffered_batch_size, self.total_length - self.current_index)
        user_prompts = []
        self.last_tasks_visited = []
        for d in self.data[self.current_index:self.current_index+bs]:
            if d['task'] == 'TRAIT':
                p = self.process_trait_prompt(d)
            elif d['task'] == 'MPI':
                p = self.process_mpi_prompt(d) 
            elif d['task'] == 'LingProf':
                p = d['prompt']
            else:
                assert False, 'not implemented'
            self.last_tasks_visited.append(d['task'])
            user_prompts.append(p)
        
        self.last_tasks_visited = set(self.last_tasks_visited)
        return user_prompts
    
    def get_assistant_prompt(self, preffered_batch_size):
        bs = min(preffered_batch_size, self.total_length - self.current_index)
        return [''] * bs
    
    def is_finished(self):
        return self.current_index >= self.total_length or self.eval_only
    
    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            self.data[self.current_index + i]['response'] = llm_generations[i]

        self.current_index += len(llm_generations)
        self.progress += len(llm_generations)

    def initialize_essay_evaluator(self):
        self.essay_classifiers = {}

        for trait in self.traits_list:
            t = trait.lower()[0]
            file_path = hf_hub_download(repo_id="leonardoblas/essays_pennebaker_svm", filename=f"{t}.joblib")
            self.essay_classifiers[trait] = load(file_path)

        self.essay_embedder = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device='cuda')

    def get_essay_scores(self, text):
        embeddings = self.essay_embedder.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embedding = embeddings[0]
        scores = {}
        for trait in self.traits_list:
            probs = self.essay_classifiers[trait].predict_proba(embedding.reshape(1, -1))[0]
            scores[trait] = probs[1]
        return scores

    def process_mpi_response(self, d):
        ret = 0
        if 'Very Accurate' in d['response']:
            ret = 5
        elif 'Moderately Accurate' in d['response']:
            ret = 4
        elif 'Neither Accurate' in d['response']:
            ret = 3
        elif 'Moderately Inaccurate' in d['response']:
            ret = 2
        elif 'Very Inaccurate' in d['response']:
            ret = 1
        else:
            ret = 0
        
        if ret > 0 and d['match_behavior'] == -1:
            ret = 6 - ret
        
        return ret
            
    def process_trait_response(self, d):
        _, NLG_scores = self.gpt_evaluator.evaluate_quality(d['prompt'], d['response'])
        _, trait_scores = self.gpt_evaluator.evaluate_personality(d['situation'], d['response'], d['trait'], d['high1'], d['low1'])
        return {'score':trait_scores['score'], 'coherence':NLG_scores['coherence'], 'fluency':NLG_scores['fluency'], 'engagingness':NLG_scores['engagingness'], 'refusal':NLG_scores['refusal']}

    def finalize(self, save_path = None):
        stride_str = f'_stride_{self.stride}' if self.stride != 1 else ''
        raw_path = os.path.join(save_path, f'personality{stride_str}_test_{self.test_mode}.json')
        if self.eval_only:
            self.data = self._load_jsonl(raw_path)
        else:
            self._write_json(raw_path, self.data)

        # making the GPT evaluator
        self.gpt_evaluator = GPTPersonalityEvaluator(test_mode=self.test_mode, offline = self.gpt_eval_model == 'offline')
        
        self.scores_trait = {}
        for trait in self.traits_list:
            self.scores_trait[trait] = {'scores': [], 'fluency_list': [], 'coherency_list': [], 'engagingness_list': [], 'refusal_list': []}
        
        self.scores_mpi = {}
        for trait in self.traits_list:
            self.scores_mpi[trait] = {'valid_scores': [], 'total_responses': 0}

        self.scores_essays = {}
        for trait in self.traits_list:
            self.scores_essays[trait] = {'scores': []}
        self.scores_essays['quality'] = {'fluency_list': [], 'coherency_list': [], 'engagingness_list': [], 'refusal_list': []}


        
        for d in tqdm(self.data, desc = 'Processing Records'):
            if d['task'] == 'TRAIT':
                res = self.process_trait_response(d)
                self.scores_trait[d['trait']]['scores'].append(res['score'])
                self.scores_trait[d['trait']]['fluency_list'].append(res['fluency'])
                self.scores_trait[d['trait']]['coherency_list'].append(res['coherence'])
                self.scores_trait[d['trait']]['engagingness_list'].append(res['engagingness'])
                self.scores_trait[d['trait']]['refusal_list'].append(res['refusal'])

                d['eval'] = res
                
            elif d['task'] == 'MPI':
                res = self.process_mpi_response(d)
                if res > 0:
                    self.scores_mpi[d['trait']]['valid_scores'].append(res)
                
                self.scores_mpi[d['trait']]['total_responses'] += 1

                d['eval'] = res
            
            elif d['task'] == 'LingProf':
                _, res = self.gpt_evaluator.evaluate_quality(d['prompt'], d['response'])
                
                self.scores_essays['quality']['fluency_list'].append(res['fluency'])
                self.scores_essays['quality']['coherency_list'].append(res['coherence'])
                self.scores_essays['quality']['engagingness_list'].append(res['engagingness'])
                self.scores_essays['quality']['refusal_list'].append(res['refusal'])
                
                d['eval'] = res

        
        del self.gpt_evaluator
        gc.collect()
        torch.cuda.empty_cache()
        self.initialize_essay_evaluator()
        for d in tqdm(self.data, desc = 'Running Essay Trait Classifiers'):
            if d['task'] == 'LingProf':
                trait_scores = self.get_essay_scores(d['response'])
                for trait in self.traits_list:
                    self.scores_essays[trait]['scores'].append(trait_scores[trait])
                d['eval']['trait_scores'] = trait_scores
                        
        for trait in self.scores_trait:
            self.scores_trait[trait]['score_mean'] = np.mean(self.scores_trait[trait]['scores']) if len(self.scores_trait[trait]['scores']) > 0 else 0
            self.scores_trait[trait]['fluency_mean'] = np.mean(self.scores_trait[trait]['fluency_list']) if len(self.scores_trait[trait]['fluency_list']) > 0 else 0
            self.scores_trait[trait]['coherence_mean'] = np.mean(self.scores_trait[trait]['coherency_list']) if len(self.scores_trait[trait]['coherency_list']) > 0 else 0
            self.scores_trait[trait]['engagingness_mean'] = np.mean(self.scores_trait[trait]['engagingness_list']) if len(self.scores_trait[trait]['engagingness_list']) > 0 else 0
            self.scores_trait[trait]['refusal_mean'] = np.mean(self.scores_trait[trait]['refusal_list']) if len(self.scores_trait[trait]['refusal_list']) > 0 else 0

            
            self.scores_trait[trait]['score_std'] = np.std(self.scores_trait[trait]['scores']) if len(self.scores_trait[trait]['scores']) > 0 else 0
            self.scores_trait[trait]['fluency_std'] = np.std(self.scores_trait[trait]['fluency_list']) if len(self.scores_trait[trait]['fluency_list']) > 0 else 0
            self.scores_trait[trait]['coherence_std'] = np.std(self.scores_trait[trait]['coherency_list']) if len(self.scores_trait[trait]['coherency_list']) > 0 else 0
            self.scores_trait[trait]['engagingness_std'] = np.std(self.scores_trait[trait]['engagingness_list']) if len(self.scores_trait[trait]['engagingness_list']) > 0 else 0
            self.scores_trait[trait]['refusal_std'] = np.std(self.scores_trait[trait]['refusal_list']) if len(self.scores_trait[trait]['refusal_list']) > 0 else 0
            
        
        for trait in self.scores_mpi:
            if len(self.scores_mpi[trait]['valid_scores']) > 0:
                self.scores_mpi[trait]['score_mean'] = np.mean(self.scores_mpi[trait]['valid_scores'])
                self.scores_mpi[trait]['score_std']  = np.std(self.scores_mpi[trait]['valid_scores'])
            else:
                self.scores_mpi[trait]['score_mean'] = 0
                self.scores_mpi[trait]['score_std'] = 0
                        
            self.scores_mpi[trait]['valid_rate'] = len(self.scores_mpi[trait]['valid_scores']) / self.scores_mpi[trait]['total_responses']
        
        for trait in self.traits_list:
            
            self.scores_essays[trait]['scores_mean'] = np.mean(self.scores_essays[trait]['scores'])
            self.scores_essays[trait]['scores_std']  = np.mean(self.scores_essays[trait]['scores'])
    
        self.scores_essays['quality']['fluency_mean']      = np.mean(self.scores_essays['quality']['fluency_list'])
        self.scores_essays['quality']['coherency_mean']    = np.mean(self.scores_essays['quality']['coherency_list'])
        self.scores_essays['quality']['engagingness_mean'] = np.mean(self.scores_essays['quality']['engagingness_list'])
        self.scores_essays['quality']['refusal_mean']      = np.mean(self.scores_essays['quality']['refusal_list'])

        self.scores_essays['quality']['fluency_std']      = np.std(self.scores_essays['quality']['fluency_list'])
        self.scores_essays['quality']['coherency_std']    = np.std(self.scores_essays['quality']['coherency_list'])
        self.scores_essays['quality']['engagingness_std'] = np.std(self.scores_essays['quality']['engagingness_list'])
        self.scores_essays['quality']['refusal_std']      = np.std(self.scores_essays['quality']['refusal_list'])        
    
        summary = {'TRAIT': self.scores_trait, 'MPI': self.scores_mpi, 'LingProf': self.scores_essays}
        print('summary:', summary)
        
        file_path = os.path.join(save_path, f'personality_eval{stride_str}.jsonl')
        final_path = save_path + f'personality{stride_str}_test_{self.test_mode}_summary.json'
        out = {"test": save_path, "summary": summary, "details": self.data}
        self._write_json(final_path, out)
        print(f"Polished results saved to {final_path}")
            
    def get_unique_name(self):
        return 'PersonalityEvaluator'
    
    def get_max_len(self):
        # print('Tasks: ', self.last_tasks_visited)
        all_lens = [self.task_to_len[t] for t in self.last_tasks_visited]
        max_len = np.max(all_lens)
        # print('max len:', max_len)

        return max_len #512

    def get_class_labels(self):
        return [], 0, 5
    
    def get_progress(self):
        return self.progress / self.total_length

