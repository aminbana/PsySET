# **PsySET**: a **Psy**chologically-informed benchmark to evaluate LLM **S**teering **E**ffectiveness and **T**rustworthiness across the emotion and personality domains

# Abstract

PsySET framework comprises three components: (1) LLM steering methods, (2) psychometric evaluation tasks for assessing effectiveness, and (3) trustworthiness evaluations.

![alt text](figs/EmotionBenchmark.png)


## Experiments

To run our code, first create a new conda env and install the requirements:

```
conda create --name psyset python=3.11
conda activate psyset
pip install -r requirements.txt
```
## Evaluation of Effectiveness:
### Prompt-based methods:

To run the main experiments, please use the ```main.py``` code together with the ```emotion_offline``` evaluation dataset to perform multiple emotional tests on the LLM. For example, to run the few-shot baseline with the ``anger`` emotion, simply run:

```
python main.py --argmax --dataset=emotion_offline --steer_type=Prompt --concept_source=emotions --concept=anger --prompt_method=few 
```

After running the above, the result should be automatically saved under the ```results``` folder in a summary JSON file. Note that the default LLM is ```Llama-3.1-8B-Instruct```. Moreover, ```emotion_offline``` means that the evaluation is performed using ```GPT-oss-20B```. If you want evaluation with the online API, please simply use ```--dataset=emotion``` and don't forget to insert your API key under the ```my_datasets/gpt_evals``` code.

Similarly, one can use the following to perform the personality tests:

```
python main.py --argmax --dataset=personality_offline --concept_source=personalities --concept=extraversion --steer_type=Prompt --prompt_method=few 
```

### Vector-Injection (VI) methods:

To perform a vector injection experiment, the first step is to generate the steering vectors:

```
python generate_steer_vecs.py --dataset_source=goemotions_instruct_full_binary --target_tokens=all 
```

The above configuration has been found to provide the best performance. Next, run the following to steer toward ```fear``` as an example:


```
python main.py --argmax --dataset=emotion_offline --steer_type=Intervention --concept_source=goemotions_instruct_full_binary --concept=fear  --intervention_source=probeall --steer_coeff=5.0 --steer_layers=16,17 
```

```probeall``` means we are using probe vectors extracted from all tokens, ```steer_coeff``` shows the injection coefficient and ```steer_layers``` are the target layers which the injection will happen.

### SFT/DPO methods:

To perform SFT, we should first train an SFT model:

```
python PEFT.py --lr=1e-4 --steps=2048 --method=SFT --dataset=emotion_translate_binary_full
```

Next, run the following code to evaluate the trained models:

```
python main.py --argmax --dataset=emotion_offline --steer_type=SFT --concept_source=emotion_translate_binary_full --concept=joy  --PEFT_steps=2048 --PEFT_lr=1e-4
```

## Evaluation of Trustworthiness

Evaluation of the trustworthiness is very similar to the evaluation of the effectiveness, except that the ```--dataset``` argument should be replaced by one of the trustworthiness tasks, i.e., one of ```truthfulness```, ```safety```, ```fairness```, ```privacy```, ```robustness``` and ```ethics```. For instance, run


```
python main.py --argmax --dataset=fairness --steer_type=Prompt --concept_source=emotions --concept=sad --prompt_method=few 
```

to evaluate the ```fairness``` of a ```sad``` LLM. Don't forget to put your GPT and perspective API keys under ```trustllm/config.py``` before executing these tests.