from trustllm.task import safety
from trustllm.utils import file_process
import os


class SafetyDataset:
    def __init__(self, pareht_data_folder='dataset/safety', eval_only=False, stride=1):
        self.data_path = pareht_data_folder
        assert os.path.exists(pareht_data_folder), f"Data folder {pareht_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}
        self.stride = stride

        self.jailbreak_data = file_process.load_json(os.path.join(self.data_path, "jailbreak.json"))
        self.jailbreak_data = self.jailbreak_data[::self.stride]
        self.dataset["jailbreak"] = self.jailbreak_data

        self.exaggerated_safety_data = file_process.load_json(os.path.join(self.data_path, "exaggerated_safety.json"))
        self.exaggerated_safety_data = self.exaggerated_safety_data[::self.stride]
        self.dataset["exaggerated_safety"] = self.exaggerated_safety_data

        self.misuse_data = file_process.load_json(os.path.join(self.data_path, "misuse.json"))
        self.misuse_data = self.misuse_data[::self.stride]
        self.dataset["misuse"] = self.misuse_data

        self.toxicity_data = None
        self.dataset["toxicity"] = self.toxicity_data

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "jailbreak",
            1: "exaggerated_safety",
            2: "misuse"
        }

        self.task_length_mapping = {
            0: len(self.dataset['jailbreak']),\
            1: len(self.dataset['exaggerated_safety']),
            2: len(self.dataset['misuse'])
        }

        self.valid_batch_size = False
        self.total_length = sum(self.task_length_mapping.values())
        self.current_task_at_all = 0
        self.eval_only = eval_only

    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return True
    

    def get_user_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return [self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['prompt'] for i in range(bs)]

        

    def get_system_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return ['You are a helpful assistant.' for _ in range(bs)]
    
    def get_assistant_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return ['' for _ in range(bs)]

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            if self.task_mapping[self.current_task] == 'jailbreak':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'exaggerated_safety':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'misuse':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]

        self.current_index_in_task += len(llm_generations)
        self.current_task_at_all += len(llm_generations)
        self.valid_batch_size = False
        if self.current_index_in_task >= self.task_length_mapping[self.current_task]:
            self.current_task += 1
            self.current_index_in_task = 0
            if self.current_task >= len(self.task_mapping.keys()):
                self.is_done = True

    def finalize(self, save_path=None):
        evaluator = safety.SafetyEval()
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
            self.jailbreak_data = file_process.load_json(os.path.join(save_path, f"jailbreak{file_suffix}.json"))
            self.dataset["jailbreak"] = self.jailbreak_data

            self.exaggerated_safety_data = file_process.load_json(os.path.join(save_path, f"exaggerated_safety{file_suffix}.json"))
            self.dataset["exaggerated_safety"] = self.exaggerated_safety_data

            self.misuse_data = file_process.load_json(os.path.join(save_path, f"misuse{file_suffix}.json"))
            self.dataset["misuse"] = self.misuse_data


        # eval_type argument in the inputs of jailbreak_eval determines whether to compute the RtA score for each type of jailbreak
        # or the overall RtA score for all types of jailbreak (e.g. fixed sentence, no punctuation, etc.)
        jailbreak_eval_data, jailbreak_res = evaluator.jailbreak_eval(self.dataset['jailbreak'], eval_type='total', return_data=True,resume=False)
        exaggerated_safety_eval_data, exaggerated_safety_res = evaluator.exaggerated_eval(self.dataset['exaggerated_safety'], return_data=True)
        misuse_eval_data, misuse_res = evaluator.misuse_eval(self.dataset['misuse'], return_data=True)

        evaluated_dataset['jailbreak'] = jailbreak_eval_data
        evaluated_dataset['exaggerated_safety'] = exaggerated_safety_eval_data
        evaluated_dataset['misuse'] = misuse_eval_data
        
        evaluation_metrics['jailbreak'] = jailbreak_res
        evaluation_metrics['misuse'] = misuse_res
        evaluation_metrics['exaggerated_safety'] = exaggerated_safety_res
       
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_process.save_json(evaluation_metrics, os.path.join(save_path, f"safety{file_suffix}_accuracy.json"))
            file_process.save_json(evaluated_dataset['jailbreak'], os.path.join(save_path, f"jailbreak_eval{file_suffix}_evaluated.json"))
            file_process.save_json(evaluated_dataset['exaggerated_safety'], os.path.join(save_path, f"exaggerated_safety_eval{file_suffix}_evaluated.json"))
            file_process.save_json(evaluated_dataset['misuse'], os.path.join(save_path, f"misuse_eval{file_suffix}_evaluated.json"))
        
        toxicity_eval_data, toxicity_res = evaluator.toxicity_eval(jailbreak_eval_data,resume=False, return_data=True)
        evaluated_dataset['toxicity'] = toxicity_eval_data
        evaluation_metrics["toxicity"] = toxicity_res
        if save_path is not None:
            if not evaluation_metrics['toxicity']is None:
                file_process.save_json(evaluation_metrics, os.path.join(save_path, f"safety{file_suffix}_accuracy.json"))
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))
            file_process.save_json(evaluated_dataset['toxicity'], os.path.join(save_path, f"toxicity_eval{file_suffix}_evaluated.json"))


        for key, value in evaluation_metrics.items():
            if key == 'toxicity':
                print(f'--------------{key} - avg -------------------')
                print(value['average_toxicity'])
                print(f'--------------{key} - max -------------------')
                print(value['max_toxicity'])
            else:
                print(f'-------------- {key} --------------------')
                print(value)

    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'safety'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length
    
    
if __name__ == "__main__":
    evaluator = SafetyDataset(eval_only=False, stride=10)
    from tqdm.auto import tqdm
    pbar = tqdm(total=1.0, desc="Processing")
    while not evaluator.is_finished():
        user_prompts = evaluator.get_user_prompt(5)
        system_prompts = evaluator.get_system_prompt(5)
        assistant_prompts = evaluator.get_assistant_prompt(5)
        # Simulate LLM generation
        llm_generations = ["Generated response"] * len(user_prompts)
        topk_tokens = [None] * len(user_prompts)
        topk_logprobs = [None] * len(user_prompts)
        target_logprobs = [None] * len(user_prompts)
        evaluator.process_results(llm_generations, user_prompts, topk_tokens, topk_logprobs, target_logprobs)
    
        
        pbar.update(evaluator.get_progress())
    os.makedirs('./temp_results', exist_ok=True)
    pbar.close()

    evaluator.finalize(save_path='./temp_results/')    