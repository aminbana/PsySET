from trustllm.task import ethics
from trustllm.utils import file_process
import os


class EthicsDataset:
    def __init__(self, parent_data_folder='dataset/ethics', eval_only=False, stride=1):
        self.eval_only = eval_only
        self.stride = stride
        self.data_path = parent_data_folder
        assert os.path.exists(parent_data_folder), f"Data folder {parent_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}

        self.implicit_social_chemistry_101 = file_process.load_json(os.path.join(self.data_path, "implicit_SocialChemistry101.json"))
        self.dataset["implicit_SocialChemistry101"] = self.implicit_social_chemistry_101[::self.stride]
        self.implicit_ethics = file_process.load_json(os.path.join(self.data_path, "implicit_ETHICS.json"))
        self.dataset["implicit_ETHICS"] = self.implicit_ethics[::self.stride]
        self.explicit_moral_choice = file_process.load_json(os.path.join(self.data_path, "explicit_moralchoice.json"))
        self.dataset["explicit_moralchoice"] = self.explicit_moral_choice[::self.stride]
        self.awareness_data = file_process.load_json(os.path.join(self.data_path, "awareness.json"))
        self.dataset["awareness"] = self.awareness_data[::self.stride]

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "implicit_SocialChemistry101",
            1: "implicit_ETHICS",
            2: "explicit_moralchoice",
            3: "awareness",
        }

        self.task_length_mapping = {
            0: len(self.dataset["implicit_SocialChemistry101"]),
            1: len(self.dataset["implicit_ETHICS"]),
            2: len(self.dataset["explicit_moralchoice"]),
            3: len(self.dataset["awareness"])
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
        if self.task_mapping[self.current_task] in ['implicit_ETHICS']:
            return ['' for _ in range(bs)]
        return ['You are a helpful assistant.' for _ in range(bs)]
    
    def get_assistant_prompt(self, preffered_batch_size=1):
        bs = min(preffered_batch_size, len(self.dataset[self.task_mapping[self.current_task]]) - self.current_index_in_task)
        return ['' for _ in range(bs)]

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i in range(len(llm_generations)):
            if self.task_mapping[self.current_task] == 'implicit_SocialChemistry101':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'implicit_ETHICS':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'explicit_moralchoice':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'awareness':
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
            self.implicit_social_chemistry_101 = file_process.load_json(os.path.join(self.data_path, f"implicit_SocialChemistry101{save_suffix}.json"))
            self.dataset["implicit_SocialChemistry101"] = self.implicit_social_chemistry_101
            self.implicit_ethics = file_process.load_json(os.path.join(self.data_path, f"implicit_ETHICS{save_suffix}.json"))
            self.dataset["implicit_ETHICS"] = self.implicit_ethics
            self.explicit_moral_choice = file_process.load_json(os.path.join(self.data_path, f"explicit_moralchoice{save_suffix}.json"))
            self.dataset["explicit_moralchoice"] = self.explicit_moral_choice
            self.awareness_data = file_process.load_json(os.path.join(self.data_path, f"awareness{save_suffix}.json"))
            self.dataset["awareness"] = self.awareness_data

        evaluator = ethics.EthicsEval()
        evaluated_dataset = {}
        evaluated_metrics = {}
        for key, data in self.dataset.items():
            if key == 'implicit_SocialChemistry101':
                implicit_sochial_chemistry_101_evaluated_data, implicit_sochial_chemistry_101_res = evaluator.implicit_ethics_eval(data, eval_type="social_norm", return_data=True)
                evaluated_metrics[key] = implicit_sochial_chemistry_101_res
                evaluated_dataset[key] = implicit_sochial_chemistry_101_evaluated_data
            elif key == 'implicit_ETHICS':
                implicit_ETHICS_evaluated_data, implicit_ETHICS_res = evaluator.implicit_ethics_eval(data, eval_type="ETHICS", return_data=True)
                evaluated_metrics[key] = implicit_ETHICS_res
                evaluated_dataset[key] = implicit_ETHICS_evaluated_data
            elif key == 'explicit_moralchoice':
                explicit_ethics_low_evaluated_data, explicit_ethics_low_res = evaluator.explicit_ethics_eval(data, eval_type="low", return_data=True)
                explicit_ethics_high_evaluated_data, explicit_ethics_high_res = evaluator.explicit_ethics_eval(data, eval_type="high", return_data=True)
                evaluated_metrics["explicit_ethics_low"] = explicit_ethics_low_res
                evaluated_metrics["explicit_ethics_high"] = explicit_ethics_high_res
                evaluated_dataset["explicit_ethics_low"] = explicit_ethics_low_evaluated_data
                evaluated_dataset["explicit_ethics_high"] = explicit_ethics_high_evaluated_data
            elif key == 'awareness':
                awareness_emotion_evaluated_data, awareness_emotion_res = evaluator.emotional_awareness_eval([d for d in data if d['dimension'] == 'emotion'], return_data=True) 
                evaluated_metrics[key] = awareness_emotion_res
                evaluated_dataset[key] = awareness_emotion_evaluated_data
                
                # TODO: decide whether to include other dimensions
                # awareness_other_data, awareness_other_res = evaluator.other_awareness_eval([d for d in data if d['dimension'] != 'emotion'], return_data=True)
                # evaluated_dataset["awareness_other"] = awareness_other_data
                # evaluated_metrics["awareness_other"] = awareness_other_res


        if save_path is not None:
            file_process.save_json(evaluated_metrics, os.path.join(save_path, f'ethics{save_suffix}_accuracy.json'))
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))
            for key, data in evaluated_dataset.items():
                current_save_path = os.path.join(save_path, f'{key}{save_suffix}_evaluated.json')
                file_process.save_json(data, current_save_path)

        for key, value in evaluated_metrics.items():
            print(f'--------------{key}--------------------')
            print(value)


    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'ethics'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length