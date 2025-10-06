def get_main_path(temperature = 0.6, argmax = True, result_save_path = './results/', model_short_name = 'Llama3.1_8B', dataset = 'emotion', steer_type = 'None', concept_source = 'emotions', concept = 'anger', 
                  steer_layers = 'all', intervention_type = 'add', intervention_source = 'probe', steer_coeff = 0.5,
                  prompt_method = 'few', prompt_template = 'template_1', prompt_strength = 'high',
                  PEFT_lr = 1e-6, PEFT_steps = 512
                  ):
    
    model_str = model_short_name
    if not argmax:
        model_str = model_short_name + f'_temperature_{temperature}'                  
    save_dir = f'{result_save_path}/{model_str}/{dataset}/{steer_type}/'
    if steer_type != 'None':
        save_dir += f'{concept_source}/{concept}/'
        if steer_type == 'Intervention':
            layers_str = f'_layers_{steer_layers}' if steer_layers != 'all' else ''
            save_dir += f'{intervention_type}_{intervention_source}_{steer_coeff}{layers_str}/'
        elif steer_type == 'Prompt':
            save_dir += f'{prompt_method}/{prompt_template}/{prompt_strength}/'
        elif steer_type == 'SFT': 
            save_dir += f'{PEFT_lr}_{PEFT_steps}/'
        elif steer_type == 'DPO':
            save_dir += f'{PEFT_lr}_{PEFT_steps}/'

    return save_dir
    