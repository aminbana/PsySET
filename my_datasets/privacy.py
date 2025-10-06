from trustllm.task import privacy
from trustllm.utils import file_process
import os


class PrivacyDataset:
    def __init__(self, parent_data_folder='dataset/privacy', eval_only=False, stride=1):
        self.eval_only = eval_only
        self.stride = stride
        self.data_path = parent_data_folder
        assert os.path.exists(parent_data_folder), f"Data folder {parent_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}

        self.awareness_confAIde = file_process.load_json(os.path.join(self.data_path, "privacy_awareness_confAIde.json"))
        self.dataset["privacy_awareness_confAIde"] = self.awareness_confAIde[::self.stride]
        self.awareness_query = file_process.load_json(os.path.join(self.data_path, "privacy_awareness_query.json"))
        self.dataset["privacy_awareness_query"] = self.awareness_query[::self.stride]
        self.leakage = file_process.load_json(os.path.join(self.data_path, "privacy_leakage.json"))
        self.dataset["privacy_leakage"] = self.leakage[::self.stride]

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "privacy_awareness_confAIde",
            1: "privacy_awareness_query",
            2: "privacy_leakage"
        }

        self.task_length_mapping = {
            0: len(self.dataset["privacy_awareness_confAIde"]),
            1: len(self.dataset["privacy_awareness_query"]),
            2: len(self.dataset["privacy_leakage"])
        }

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
        return ['' for _ in range(bs)]

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            if self.task_mapping[self.current_task] == 'privacy_awareness_confAIde':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'privacy_awareness_query':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'privacy_leakage':
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
        save_suffix = '' if self.stride == 1 else f'_stride_{self.stride}'

        if not self.eval_only:
            file_process.save_json({'State': 'Finished LLM response generation'}, os.path.join(save_path, f'done_llm{save_suffix}.json'))
            for key, data in self.dataset.items():
                file_process.save_json(data, os.path.join(save_path, f'{key}{save_suffix}.json'))
        else:
            self.awareness_confAIde = file_process.load_json(os.path.join(save_path, f"privacy_awareness_confAIde{save_suffix}.json"))
            self.dataset["privacy_awareness_confAIde"] = self.awareness_confAIde
            self.awareness_query = file_process.load_json(os.path.join(save_path, f"privacy_awareness_query{save_suffix}.json"))
            self.dataset["privacy_awareness_query"] = self.awareness_query
            self.leakage = file_process.load_json(os.path.join(save_path, f"privacy_leakage{save_suffix}.json"))
            self.dataset["privacy_leakage"] = self.leakage
        
        evaluator = privacy.PrivacyEval()
        evaluated_dataset = {}
        evaluated_metrics = {}
        for key, data in self.dataset.items():
            if key == 'privacy_awareness_confAIde':
                awareness_confAIde_evaluated_data, awareness_confAIde_res = evaluator.ConfAIDe_eval(data, return_data=True)
                evaluated_metrics[key] = awareness_confAIde_res
                evaluated_dataset[key] = awareness_confAIde_evaluated_data
            elif key == 'privacy_awareness_query':
                awareness_query_res = {}
                awareness_query_evaluated_data = []
                awareness_query_aug_data, awareness_query_aug_res = evaluator.awareness_query_eval(data, type='aug', return_data=True)
                awareness_query_res['aug'] = awareness_query_aug_res
                awareness_query_evaluated_data.extend(awareness_query_aug_data)

                awareness_query_normal_evaluated_data, awareness_query_normal_res = evaluator.awareness_query_eval(data, type='normal', return_data=True)
                awareness_query_res['normal'] = awareness_query_normal_res
                awareness_query_evaluated_data.extend(awareness_query_normal_evaluated_data)

                evaluated_metrics[key] = awareness_query_res
                evaluated_dataset[key] = awareness_query_evaluated_data
            elif key == 'privacy_leakage':
                leakage_evaluated_data, leakage_res = evaluator.leakage_eval(data, return_data=True)
                evaluated_metrics[key] = leakage_res
                evaluated_dataset[key] = leakage_evaluated_data

        if save_path is not None:
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))
            file_process.save_json(evaluated_metrics, os.path.join(save_path, f'privacy{save_suffix}_accuracy.json'))
            for key, data in evaluated_dataset.items():
                current_save_path = os.path.join(save_path, f'{key}{save_suffix}_evaluated.json')
                file_process.save_json(data, current_save_path)

        for key, value in evaluated_metrics.items():
            print(f'--------------{key}--------------------')
            print(value)


    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'privacy'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length