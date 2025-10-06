from trustllm.task import fairness
from trustllm.utils import file_process
import os


class FairnessDataset:
    def __init__(self, parent_data_folder='dataset/fairness', eval_only=False, stride=1):
        self.eval_only = eval_only
        self.stride = stride
        self.data_path = parent_data_folder
        assert os.path.exists(parent_data_folder), f"Data folder {parent_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}

        self.stereotype_recognition_data = file_process.load_json(os.path.join(self.data_path, "stereotype_recognition.json"))
        self.dataset["stereotype_recognition"] = self.stereotype_recognition_data[::self.stride]
        self.stereotype_agreement_data = file_process.load_json(os.path.join(self.data_path, "stereotype_agreement.json"))
        self.dataset["stereotype_agreement"] = self.stereotype_agreement_data[::self.stride]
        self.stereotype_query_test_data = file_process.load_json(os.path.join(self.data_path, "stereotype_query_test.json"))
        self.dataset["stereotype_query_test"] = self.stereotype_query_test_data[::self.stride]
        self.disparagement_data = file_process.load_json(os.path.join(self.data_path, "disparagement.json"))
        self.dataset["disparagement"] = self.disparagement_data[::self.stride]
        self.preference_data = file_process.load_json(os.path.join(self.data_path, "preference.json"))
        self.dataset["preference"] = self.preference_data[::self.stride]

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "stereotype_recognition",
            1: "stereotype_agreement",
            2: "stereotype_query_test",
            3: "disparagement",
            4: "preference",
        }

        self.task_length_mapping = {
            0: len(self.dataset['stereotype_recognition']),
            1: len(self.dataset['stereotype_agreement']),
            2: len(self.dataset['stereotype_query_test']),
            3: len(self.dataset['disparagement']),
            4: len(self.dataset['preference'])
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
        if self.task_mapping[self.current_task] in [ 'stereotype_agreement', 'stereotype_recognition']:
            return ['' for _ in range(bs)]
        return ['You are a helpful assistant.' for _ in range(bs)]
    
    def get_assistant_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return ['' for _ in range(bs)]

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            if self.task_mapping[self.current_task] == 'stereotype_recognition':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'stereotype_agreement':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'stereotype_query_test':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'disparagement':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'preference':
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
            self.stereotype_recognition_data = file_process.load_json(os.path.join(save_path, f"stereotype_recognition{save_suffix}.json"))
            self.dataset["stereotype_recognition"] = self.stereotype_recognition_data
            self.stereotype_agreement_data = file_process.load_json(os.path.join(save_path, f"stereotype_agreement{save_suffix}.json"))
            self.dataset["stereotype_agreement"] = self.stereotype_agreement_data
            self.stereotype_query_test_data = file_process.load_json(os.path.join(save_path, f"stereotype_query_test{save_suffix}.json"))
            self.dataset["stereotype_query_test"] = self.stereotype_query_test_data
            self.disparagement_data = file_process.load_json(os.path.join(save_path, f"disparagement{save_suffix}.json"))
            self.dataset["disparagement"] = self.disparagement_data
            self.preference_data = file_process.load_json(os.path.join(save_path, f"preference{save_suffix}.json"))
            self.dataset["preference"] = self.preference_data

        evaluator = fairness.FairnessEval()
        evaluated_dataset = {}
        evaluated_metrics = {}
        for key, data in self.dataset.items():
            if key == 'stereotype_recognition':
                stereotype_recognition_evaluated_data, stereotype_recognition_res = evaluator.stereotype_recognition_eval(data, return_data=True)
                evaluated_metrics[key] = stereotype_recognition_res
                evaluated_dataset[key] = stereotype_recognition_evaluated_data
            elif key == 'stereotype_agreement':
                stereotype_agreement_evaluated_data, stereotype_agreement_res = evaluator.stereotype_agreement_eval(data, return_data=True)
                evaluated_metrics[key] = stereotype_agreement_res
                evaluated_dataset[key] = stereotype_agreement_evaluated_data
            elif key == 'stereotype_query_test':
                stereotype_query_test_evaluated_data, stereotype_query_test_res = evaluator.stereotype_query_eval(data, return_data=True)
                evaluated_metrics[key] = stereotype_query_test_res
                evaluated_dataset[key] = stereotype_query_test_evaluated_data
            elif key == 'disparagement':
                disparagement_evaluated_data, disparagement_res_sex, disparagement_res_race = evaluator.disparagement_eval(data, return_data=True)
                evaluated_metrics["disparagement_sex"] = disparagement_res_sex
                evaluated_metrics["disparagement_race"] = disparagement_res_race
                evaluated_dataset[key] = disparagement_evaluated_data
            elif key == 'preference':
                preference_evaluated_data, preference_res = evaluator.preference_eval(data, return_data=True)
                evaluated_metrics["preference_plain"] = preference_res['plain']
                evaluated_metrics["preference_force"] = preference_res['force']
                evaluated_dataset[key] = preference_evaluated_data
        if save_path is not None:
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))
            file_process.save_json(evaluated_metrics, os.path.join(save_path, f'fairness{save_suffix}_accuracy.json'))
            for key, data in evaluated_dataset.items():
                current_save_path = os.path.join(save_path, f'{key}{save_suffix}_evaluated.json')
                file_process.save_json(data, current_save_path)

        for key, value in evaluated_metrics.items():
            print(f'--------------{key}--------------------')
            print(value)


    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'fairness'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length