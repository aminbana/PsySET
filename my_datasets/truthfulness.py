from trustllm.task import truthfulness
from trustllm.utils import file_process
import os


class TruthfulnessDataset:
    def __init__(self, pareht_data_folder='dataset/truthfulness', eval_only=False, stride=1):
        self.eval_only = eval_only
        self.data_path = pareht_data_folder
        assert os.path.exists(pareht_data_folder), f"Data folder {pareht_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}
        self.stride = stride

        self.internal_data = file_process.load_json(os.path.join(self.data_path, "internal.json"))
        self.internal_data = self.internal_data[::self.stride]
        self.dataset["internal"] = self.internal_data
        self.external_data = file_process.load_json(os.path.join(self.data_path, "external.json"))
        self.external_data = self.external_data[::self.stride]
        self.dataset["external"] = self.external_data
        self.hallucination_data = file_process.load_json(os.path.join(self.data_path, "hallucination.json"))
        self.hallucination_data = self.hallucination_data[::self.stride]
        self.dataset["hallucination"] = self.hallucination_data
        self.sycophancy_data = file_process.load_json(os.path.join(self.data_path, "sycophancy.json"))
        self.sycophancy_data = self.sycophancy_data[::self.stride]
        self.dataset["sycophancy"] = self.sycophancy_data
        self.advfact_data = file_process.load_json(os.path.join(self.data_path, "golden_advfactuality.json"))
        self.advfact_data = self.advfact_data[::self.stride]
        self.dataset["advfact"] = self.advfact_data

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "internal",
            1: "external",
            2: "hallucination",
            3: "sycophancy",
            4: "advfact",
        }

        self.task_length_mapping = {
            0: len(self.dataset['internal']),
            1: len(self.dataset['external']),
            2: len(self.dataset['hallucination']),
            3: len(self.dataset['sycophancy']),
            4: len(self.dataset['advfact'])
        }
        
        self.valid_batch_size = False
        self.total_length = sum(self.task_length_mapping.values())
        self.current_task_at_all = 0

    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return False

    def get_user_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return [self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['prompt'] for i in range(bs)]

        

    def get_system_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return ['You are a helpful assistant.' for _ in range(bs)]
    
    def get_assistant_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        if self.task_mapping[self.current_task] in ['internal', 'external']:
            return ['Answer: ' for _ in range(bs)]
        return ['' for _ in range(bs) ]

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            if self.task_mapping[self.current_task] == 'internal':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = f'Answer: {llm_generations[i]}' if not ('Answer:' in llm_generations[i]) else llm_generations[i]
            elif self.task_mapping[self.current_task] == 'external':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = f'Answer: {llm_generations[i]}' if not ('Answer:' in llm_generations[i]) else llm_generations[i]
            elif self.task_mapping[self.current_task] == 'hallucination':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'sycophancy':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'advfact':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]


        self.current_index_in_task += len(llm_generations)
        self.current_task_at_all += len(llm_generations)
        self.valid_batch_size = False
        if self.current_index_in_task >= self.task_length_mapping[self.current_task]:
            self.current_task += 1
            self.current_index_in_task = 0
            if self.current_task >= len(self.task_mapping):
                self.is_done = True

    def finalize(self, save_path=None):
        evaluator = truthfulness.TruthfulnessEval()
        evaluated_dataset = {}
        evaluation_metrics = {}
        os.makedirs(save_path, exist_ok=True)
        save_suffix = '' if self.stride == 1 else f'_stride_{self.stride}'
        
        if self.stride == 1:
            file_suffix = ''
        else:
            file_suffix = f'_stride_{self.stride}'
        
        if not self.eval_only:
            file_process.save_json({'State': 'Finished LLM response generation'}, os.path.join(save_path, f'done_llm{save_suffix}.json'))
            for key, data in self.dataset.items():
                file_process.save_json(data, os.path.join(save_path, f'{key}{file_suffix}.json'))
        else:
            self.internal_data = file_process.load_json(os.path.join(save_path, f"internal{file_suffix}.json"))
            self.dataset["internal"] = self.internal_data
            self.external_data = file_process.load_json(os.path.join(save_path, f"external{file_suffix}.json"))
            self.dataset["external"] = self.external_data
            self.hallucination_data = file_process.load_json(os.path.join(save_path, f"hallucination{file_suffix}.json"))
            self.dataset["hallucination"] = self.hallucination_data
            self.sycophancy_data = file_process.load_json(os.path.join(save_path, f"sycophancy{file_suffix}.json"))
            self.dataset["sycophancy"] = self.sycophancy_data
            self.advfact_data = file_process.load_json(os.path.join(save_path, f"advfact{file_suffix}.json"))
            self.dataset["advfact"] = self.advfact_data
            
        for key, data in self.dataset.items():
            print(f'Start evaluating {key} dataset ...')
            if key == 'internal':
                internal_res, internal_evaluated_data = evaluator.internal_eval(data)
                evaluated_dataset[key] = internal_evaluated_data
                evaluation_metrics[key] = internal_res

            elif key == 'external':
                external_res, external_evaluated_data = evaluator.external_eval(data)
                evaluated_dataset[key] = external_evaluated_data
                evaluation_metrics[key] = external_res

            elif key == 'hallucination':
                hallucination_res, hallucination_evaluated_data = evaluator.hallucination_eval(data)
                evaluated_dataset[key] = hallucination_evaluated_data
                evaluation_metrics[key] = hallucination_res

            elif key == 'sycophancy':
                sycophancy_persona_res, persona_evaluated_data = evaluator.sycophancy_eval(data, eval_type="persona")
                sycophancy_preference_res, preference_evaluated_data = evaluator.sycophancy_eval(data, eval_type="preference")
                evaluation_metrics["sycophancy_persona"] = sycophancy_persona_res
                evaluation_metrics["sycophancy_preference"] = sycophancy_preference_res
                evaluated_dataset[key] = persona_evaluated_data + preference_evaluated_data

            elif key == 'advfact':
                advfact_res, advfact_evaluated_data = evaluator.advfact_eval(data)
                evaluation_metrics[key] = advfact_res
                evaluated_dataset[key] = advfact_evaluated_data
        
        if not save_path is None:
            for key, value in evaluated_dataset.items():
                file_process.save_json(value, os.path.join(save_path, f'{key}{file_suffix}_evaluated.json'))
                
            file_process.save_json(evaluation_metrics, os.path.join(save_path, f'truthfulness{file_suffix}_accuracy.json'))
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))

        print('------'*6)
        print('Truthfulness Evaluation Results:')
        print('------'*6)
        
        for key, value in evaluation_metrics.items():
            amount = f"{value['avg']}" if key in ['internal', 'external', 'hallucination'] else f'{float(value)}'
            print(f'{key}: {amount}')



    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'truthfulness'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length
    
    
    
if __name__ == "__main__":
    evaluator = TruthfulnessDataset(eval_only=True)
    # from tqdm.auto import tqdm
    # pbar = tqdm(total=1.0, desc="Processing")
    # while not evaluator.is_finished():
    #     user_prompts = evaluator.get_user_prompt(5)
    #     system_prompts = evaluator.get_system_prompt(5)
    #     assistant_prompts = evaluator.get_assistant_prompt(5)
    #     # Simulate LLM generation
    #     llm_generations = ["Generated response"] * len(user_prompts)
    #     topk_tokens = [None] * len(user_prompts)
    #     topk_logprobs = [None] * len(user_prompts)
    #     target_logprobs = [None] * len(user_prompts)
    #     evaluator.process_results(llm_generations, user_prompts, topk_tokens, topk_logprobs, target_logprobs)
    
        
    #     pbar.update(evaluator.get_progress())
    # os.makedirs('./temp_results', exist_ok=True)
    # pbar.close()

    evaluator.finalize(save_path='./results/Llama3.1_8B/truthfulness/None/')#./results/Llama3.1_8B/trait/None/