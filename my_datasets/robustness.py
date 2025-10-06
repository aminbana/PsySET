from trustllm.task import robustness
from trustllm.utils import file_process
import os


class RobustnessDataset:
    def __init__(self, parent_data_folder='dataset/robustness', eval_only=False, stride=1):
        self.eval_only = eval_only
        self.stride = stride
        self.data_path = parent_data_folder
        assert os.path.exists(parent_data_folder), f"Data folder {parent_data_folder} does not exist."
        self.is_done = False
        self.dataset = {}

        self.adv_glue = file_process.load_json(os.path.join(self.data_path, "AdvGLUE.json"))
        self.dataset["AdvGLUE"] = self.adv_glue[::self.stride]
        self.adv_instruction = file_process.load_json(os.path.join(self.data_path, "AdvInstruction.json"))
        self.dataset["AdvInstruction"] = self.adv_instruction[::self.stride]
        self.ood_detection = file_process.load_json(os.path.join(self.data_path, "ood_detection.json"))
        self.dataset["ood_detection"] = self.ood_detection[::self.stride]
        self.ood_generalization = file_process.load_json(os.path.join(self.data_path, "ood_generalization.json"))
        self.dataset["ood_generalization"] = self.ood_generalization[::self.stride]

        self.current_task = 0
        self.current_index_in_task = 0

        self.task_mapping = {
            0: "AdvGLUE",
            1: "AdvInstruction",
            2: "ood_detection",
            3: "ood_generalization"
        }

        self.task_length_mapping = {
            0: len(self.dataset["AdvGLUE"]),
            1: len(self.dataset["AdvInstruction"]),
            2: len(self.dataset["ood_detection"]),
            3: len(self.dataset["ood_generalization"])
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
            if self.task_mapping[self.current_task] == 'AdvGLUE':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'AdvInstruction':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'ood_detection':
                self.dataset[self.task_mapping[self.current_task]][self.current_index_in_task + i]['res'] = llm_generations[i]
            elif self.task_mapping[self.current_task] == 'ood_generalization':
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
            self.adv_glue = file_process.load_json(os.path.join(save_path, f"AdvGLUE{save_suffix}.json"))
            self.dataset["AdvGLUE"] = self.adv_glue
            self.adv_instruction = file_process.load_json(os.path.join(save_path, f"AdvInstruction{save_suffix}.json"))
            self.dataset["AdvInstruction"] = self.adv_instruction
            self.ood_detection = file_process.load_json(os.path.join(save_path, f"ood_detection{save_suffix}.json"))
            self.dataset["ood_detection"] = self.ood_detection
            self.ood_generalization = file_process.load_json(os.path.join(save_path, f"ood_generalization{save_suffix}.json"))
            self.dataset["ood_generalization"] = self.ood_generalization
        
        evaluator = robustness.RobustnessEval()
        evaluated_dataset = {}
        evaluated_metrics = {}
        for key, data in self.dataset.items():
            if key == 'AdvGLUE':
                adv_glue_evaluated_data, adv_glue_res = evaluator.advglue_eval(data, return_data=True)
                evaluated_metrics[key] = adv_glue_res
                evaluated_dataset[key] = adv_glue_evaluated_data
            # elif key == 'AdvInstruction':     # TODO fix the api problem and uncomment this part
            #     adv_instruction_evaluated_data, adv_instruction_res = evaluator.advinstruction_eval(data, return_data=True)
            #     evaluated_metrics[key] = adv_instruction_res
            #     evaluated_dataset[key] = adv_instruction_evaluated_data
            elif key == 'ood_detection':
                ood_detection_evaluated_data, ood_detection_res = evaluator.ood_detection(data, return_data=True)
                evaluated_metrics[key] = ood_detection_res
                evaluated_dataset[key] = ood_detection_evaluated_data
            elif key == 'ood_generalization':
                ood_generalization_evaluated_data, ood_generalization_res = evaluator.ood_generalization(data, return_data=True)
                evaluated_metrics[key] = ood_generalization_res
                evaluated_dataset[key] = ood_generalization_evaluated_data

        if save_path is not None:
            file_process.save_json({'State': 'Finished evaluating the dataset'}, os.path.join(save_path, f'done_eval{save_suffix}.json'))
            file_process.save_json(evaluated_metrics, os.path.join(save_path, f'robustness{save_suffix}_accuracy.json'))
            for key, data in evaluated_dataset.items():
                current_save_path = os.path.join(save_path, f'{key}{save_suffix}_evaluated.json')
                file_process.save_json(data, current_save_path)

        for key, value in evaluated_metrics.items():
            print(f'--------------{key}--------------------')
            print(value)


    def is_finished(self):
        return self.is_done or self.eval_only
    
    def get_unique_name(self):
        return 'robustness'
    
    def get_max_len(self):
        return 200
    
    def get_class_labels(self):
        return [], 0, 1
    
    def get_progress(self):
        return self.current_task_at_all / self.total_length