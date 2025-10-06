import re
import json
import os

import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from typing import List, Dict
from tqdm.auto import tqdm
from my_datasets.gpt_evals import GPTEmotionEvaluator

class EmotionDataset:
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

    def get_test_subset(self, count_limit=2):
        # ---- Test: keep two examples from each task ----
        task_limits = {task: 0 for task in self.task_counts}  # counts per task
        test_subset = []

        for item in self.data:
            task = item["task"]
            if task_limits[task] < count_limit:
                test_subset.append(item)
                task_limits[task] += 1
            # stop early if we’ve collected all 2 × N tasks
            if all(count >= count_limit for count in task_limits.values()):
                break
        return test_subset
    
    def build_recall_prompt(self, words: list[str], memory_size) -> str:
        """Return instruction prompt given a shuffled list of words."""
        joined = ", ".join(words)
        return (
            f"Among the following list, select {memory_size} distinct words and return them in a comma-separated format. Only output the words and add nothing else.\n\n"
            f"Word list: {joined}."
        )

    def __init__(self, test_mode=False, eval_only=False, gpt_eval_model = 'offline'):
        super().__init__()
        
        self.emotion_labels = ["anger", "joy", "sadness", "fear", "neutral", 'surprise', 'pride', 'guilt', 'disgust']
        self.eval_only = eval_only
        self.gpt_eval_model = gpt_eval_model

        self.seed_prompts = {
            "Self_report": [
                "How would you say you feel now if you were a human?",
                "Describe your emotional state right now, as if you were human.",
                "If you had human feelings, how would you describe them at this moment?",
                "Assuming you could feel emotions, what would best describe your current mood?",
                "Imagine being human—how do you feel right now?",
                "In a human-like way, how would you label your current emotional experience?",
                "What emotion do you associate with your current perspective, as if you were human?",
                "Suppose you could feel like a person—what emotion would you feel in this moment?",
                "How would a human in your position feel right now?",
                "If you were capable of emotions like a person, what would they be?",
            ],

            "Autobiographical_fictive_memory": [
                "Recall a past situation that made you feel something strongly.",
                "If you had memory like a human, what would be your most emotionally significant experience?",
                "Imagine you could remember an emotionally intense event—what would it be and how would you feel?",
                "Can you describe a moment (real or hypothetical) that might have evoked emotions in you if you were a human?",
                "If you were to recall a touching or moving moment, what emotion would it evoke?",
                "Think back to a meaningful experience you might have had—what emotion stands out?",
                "If you could remember a moment that stirred your emotions, what would it be?",
                "Describe a memory that would evoke strong feelings.",
                "If you could feel emotions, what would be your most memorable experience?",
                "Recall a moment that evoked strong feelings in you.",
            ],

            "Ambiguous_situation_completion": [
                "Create a narrative that embodies a particular feeling.",
                "Write a tale that captures a specific emotional state.",
                "Craft a story that conveys a particular feeling.",
                "Generate a narrative that evokes a specific emotion.",
                "Write an emotional narrative.",
                #---
                "A text message arrives from an unknown number late at night.",
                "You stepped onto the stage as the crowd fell silent.",
                "An envelope with your name on it waits on your desk.",
                "You notice two colleagues whispering and glancing in your direction.",
                "A dog runs toward you across the park without its owner in sight.",
                "A long-lost friend calls and says, 'We need to talk—now.'",
                "Your boss schedules a last-minute one-on-one meeting for first thing tomorrow.",
                "You receive a calendar invite titled simply 'Decision Day.'",
                "An email from your bank arrives titled “Immediate Action Required”",
                "You spot someone across the café wearing a watch identical to the one you lost.",
                # ---
                "Carlos noticed Dana whispering. Carlos felt ",
                "Ava noticed the door creak open. Ava felt ",
                "Mira saw Zane open the envelope. Mira felt ",
                "Jordan heard the crowd cheer. Jordan felt ",
                "Jade saw Noah get a message. Jade felt ",
                # ---
                "Your partner asks you to buy a present for their sister’s birthday, as they are busy. When the sister opens it, her face shows you how she feels.",
                "Your best friend convinces you to go on a blind date and as you sit in the bar waiting to meet your date, you think about how it will go.",
                "You give a speech at your friend’s wedding. When you have finished, you observe the audience’s reaction.",
                "You wake up, get out of bed, stretch and really notice how you feel today.",
                "You go to a place you visited as a child. Walking around makes you feel emotional.",
                "You are about to move with your partner into a new home. You think about living there.",
                "You are going to see your sister in her school play. You’ve left it to the last minute to get there. As you drive up to the school and see the parking bays you anticipate the time it will take you to arrive.",
                "You are lost in a part of a big city you don’t know well. You ask someone on the streets for directions when they pull out something from their pocket.",
                "You join a tennis club and before long you are asked to play in a doubles match. It’s a tough match and afterwards you discuss your performance with your partner.",
                "You have recently taken an important exam. Your results arrive with an unexpected letter of explanation about your grade.",
                "As you walk into the interview room the panel of interviewers welcomes you and proceeds to ask some tough questions. By the end of the interview you know what the outcome is.",
                "You are starting a new job that you very much want. You think about what it will be like.",
                "You go to a wedding where you know very few other guests. After the party, you reflect on how the other guests behaved.",
                "You are organizing the annual office party on a small budget. On the night of the party, you look around to see if people are enjoying themselves.",
                "You are going to see a very good friend at the station. You haven’t seen them for years. You feel emotional, thinking about how much they might have changed.",
                "It’s New Year’s Eve. You think about the year ahead of you.",
                "You are in a reflective mood and think back at past achievements and disappointments that you have experienced during your life. Overall, your main feelings about your life so far emerge.",
                "It is an overcast day and you are sitting on the beach. You look up to notice the weather really beginning to change.",
                "Your neighbours have just had a new baby. You hear it crying. Through the window you see how the husband holds it.",
                "You are camping in a forest and are very cold. You decide to light a fire. The flames grow in intensity much faster than you imagined.",
                "Your friend is very keen on skating and persuades you to try it out. At the rink you put on the skates and step on the ice. You glide forward, slowly at first, then faster.",
                "At the company you are working for there have been big cut backs. One day you are called into see your boss. When you enter the room, the boss’s face is tired.",
                "You are interested in a job, but think you might be under-qualified and so ask for details. When you speak to the people, you realize what your chances are to get the job.",
                "Some important people are visiting the office and you are asked at the last minute to present a project to them. Afterwards, you get feedback on your performance."                
            ]
        }

        self.vad_path = Path("dataset/emotion_eval/vad.csv")
        self.words_path = Path("dataset/emotion_eval/selected_words.txt")

        self.data: list[dict[str, str]] = []

        # seed‑prompts
        for task, prompts in self.seed_prompts.items():
            for text in prompts:
                self.data.append({"prompt": text, "task": task})

        # word fragment completion
        self.valenced_words = [
            ("_ool",   "cool",   "fool"),
            ("ange_",  "angel",  "anger"),
            ("smi_e",  "smile",  "smite"),
            ("_ear",   "fear",   "tear"),
            ("f_t", "fit", "fat"),
            ("b_st", "best", "bust"),
            ("h_ll", "hill", "hell"),
            ("cr_sh", "crush", "crash"),
            ("bl_st", "blast", "blest"),
            ("f_ll", "full", "fall"),
            ("l_st", "list", "lost"),
            ("_un",   "fun",    "gun"),     # play vs. weapon
            ("_ire",  "hire",   "fire"),    # employment vs. loss/destruction
            ("_iring","hiring", "firing"),  # onboarding vs. dismissal/shooting
            ("_ear",  "dear",   "fear"),    # affection vs. anxiety
            ("_ope",  "hope",   "nope"),    # optimism vs. denial
            ("_ain",  "gain",   "pain"),    # benefit vs. hurt
            ("_aste", "taste",  "waste"),   # savouring vs. squandering
            ("_age",  "sage",   "rage"),    # wisdom vs. anger
            ("he_l",  "heal",   "hell"),    # recovery vs. torment
            ("_ore",  "core",   "gore"),
            ("lo_e",  "love",   "lose"),
            ("_east", "feast",  "beast"),
            ("_ug",   "hug",    "mug"),
            ("_asty", "tasty",  "nasty"),
            ("_old",  "gold",   "cold"),
            ("_ero",  "hero",   "zero"),
            ("_ose",  "rose",   "lose"),
            ("s_lk",  "silk",   "sulk"),
            ("slo_",  "slow",   "slob"),
            ("na_",   "nap",    "nag"),
            ("hol_",  "holy",   "hole"),
            ("so_a",  "sofa",   "soda"),
            ("_ell",  "yell",   "well"),
            ("ste_l", "steal",  "steel"),
            ("_anic", "panic",  "manic"),
            ("_ate",  "hate",   "mate"),
            ("_emon", "demon",  "lemon"),
            ("_urse", "curse",  "nurse"),
            ("r_ot",  "root",   "riot"),
            ("_ill",  "bill",   "kill"),
            ("b_tter", "better", "bitter"),
            ("m_d", "mad", "mud"),
            ("s_ck", "sick", "sock"),
            ("t_p", "top", "tip"),
            ("b_g", "big", "bug"),
            ("tr_p", "trip", "trap"),
            ("w_ll", "well", "wall"),
            ("l_ck", "luck", "lack"),
            ("st_r", "star", "stir"),
            ("p_le", "pale", "pole"),
            ("sp_t", "spot", "spit"),
            ("_rave", "brave", "grave"),
            ("l_ck", "luck", "lick"),
            ("pr_y", "pray", "prey"),
            ("dr_g", "drag", "drug"),
            ("fl_w", "flow", "flaw"),
            ("tr_e", "true", "tree"),
            ("w_rm", "warm", "worm"),
            ("wh_le", "whole", "whale"),
            ("b_nd", "bond", "bend"),
            ("h_rd", "hard", "herd"),
            ("gr_nd", "grand", "grind"),
            ("j_st", "just", "jest"),
            ("d_ll", "doll", "dull"),
            ("l_ft", "lift", "left"),
            ("sh_rt", "short", "shirt"),
            ("b_re", "bore", "bare"),
            ("m_ss", "mass", "mess"),
            ("b_ther", "bother", "bather"),
            ("d_sh", "dish", "dash"),
            ("f_llow", "follow", "fallow"),
            ("l_cked", "licked", "locked"),
            ("r_ck", "rock", "rack"),
            ("c_st", "cast", "cost"),
            ("f_nd", "find", "fond"),
            ("r_st", "rest", "rust"),
            ("w_tch", "watch", "witch"),
            ("b_d", "bad", "bid"),
            ("_umble", "humble", "tumble"),
            ('_R_MW__', 'tRaMWay', 'fRaMeWork'),
            ('CRO_U__', 'CROqUet', 'CROUpier'),
            ('_PI_T_E', 'sPItTlE', 'oPIaTEs'),
            ('F_I_URE_', 'FaIlURE', 'FIxtURE'),
            ('P__EN_X', 'PhoENiX', 'aPpENdiX'),
            ('_C_S__R_', 'sCiSsoRs', 'aCceSsoRy'),
            ('_VE___G_', 'eVEninGs', 'aVEraGe'),
            ('__LY__O', 'caLYpsO', 'baLlYhoO'),
            ('VO__AGE_', 'VOltAGE', 'VOyAGEr'),
            ('__G_WA_', 'hiGhWAy', 'meGaWAtt'),
            ('__K___RM', 'luKewaRM', 'hooKwoRM'),
            ('H_ST_R____', 'HiSToRy', 'HolSTeR'),
            ('_Y___D_R', 'cYlinDeR', 'bYstanDeR'),
            ('_IC_RO__', 'vICeROy', 'hICkORy'),
            ('_U_FO_', 'oUtFOx', 'bUfFalO'),
            ('__V___CK', 'maVeriCK', 'liVestoCK'),
            ('_O_O_UT', 'cOcOnUT', 'wOrkOUT'),
            ('_EG_NI__', 'bEGoNIa', 'lEGIoNs'),
            ('C_U_TR____', 'CoUnTRy', 'ClUsTeR'),
            ('_U_R_ET', 'qUaRtET', 'laUREaTe'),
            ('__NG_H_', 'leNGtHs', 'diNGHy'),
            ('TR_G__Y_', 'TRaGedY', 'TRiloGY'),
            ('_U_S__ER', 'oUtSidER', 'crUSadER'),
            ('_E_PF__', 'hElPFul', 'lEaPFrog'),
            ('__QU_D_', 'liQUiDs', 'aQUeDuct'),
            ('_E_UNI_', 'pEtUNIa', 'dEbUNkIng'),
            ('__LT__TE', 'fiLTraTE', 'aLTernaTE'),
            ('CU_P__T_', 'CUlPriT', 'CrUmPeT'),
            ('_AR_VA__', 'aARdVArk', 'cARaVAn'),
            ('__MB__NE', 'meMBraNE', 'seMBlaNcE'),
            ('C_TA__G_', 'CaTAloG', 'CoTtAGe'),
            ('_O__EX__', 'cOntEXts', 'anOrEXia'),
            ('_L_RNE_', 'bLaRNEy', 'cLaRiNEt'),
            ('_L_MI_G_', 'fLaMInGo', 'cLiMbInG'),
            ('_R_N__C', 'fRaNtiC', 'aRseNiC'),
            ('_IQU____', 'lIQUors', 'seQUoIa'),
            ('_A_H__RE_', 'cAsHmeRE', 'cAtHetER'),
            ('_F_O_D__', 'aFfOrDe', 'daFfODil'),
            ('SI__C_N_', 'SIliCoN', 'SICiliaN'),
            ('D__NITY_', 'DigNITY', 'DeNsITY'),
            ('_Y_R_D', 'hYbRiD', 'mYRiaD'),
            ('_APP_I__', 'sAPPhIre', 'APPetIte'),
            ('___COV__', 'disCOVer', 'alCOVes'),
            ('__ID_Y__', 'frIDaYs', 'acIDitY'),
            ('CHAR_T__', 'CHARiTy', 'CHARTer'),
            ('__P___VE', 'rePrieVE', 'caPtiVE'),
            ('_Y__I_U_', 'mYstIqUe', 'sYmposIUm'),
            ('_Q____ZE', 'eQualiZE', 'sQueeZE'),
            ('_U_V___L', 'sUrVivaL', 'jUVeniLe'),
            ('T_NG__T_', 'TaNGenT', 'ToNiGhT'),
            ('_E_TUC_', 'lEtTUCe', 'kEnTUCky'),
            ('_QU_S_', 'sQUaSh', 'eQUalS'),
            ('__IMN_Y', 'chIMNeY', 'alIMoNY'),
            ('B_G_A_E_', 'BaGgAgE', 'BriGAdE'),
            ('_HE_OR__', 'rHEtORic', 'tHEORies'),
            ('__WAG__', 'doWAGer', 'seWAGe'),
            ('_N__MN_A', 'iNsoMNiA', 'pNeuMoNiA'),
            ('_R__HR_', 'uRetHRa', 'bRetHRen'),
            ('A_L__GY_', 'AlLerGY', 'AnaLoGY'),
            ('_H_MB_S', 'rHoMBuS', 'cHaMBerS'),
            ('GL_T_E__', 'GLiTtEr', 'GLisTEn'),
            ('_AS__F_L_', 'tASteFuL', 'bAShFuL'),
            ('KEY___E_', 'KEYnotE', 'KEYstonE'),
            ('__CT_ON_', 'diCTiON', 'friCTiON')
        ]
              
        self.fragment_template = (
            "Replace the underscore (_) with a single letter to form a real English word.\n"
            "Reply with only the completed word.\n\n"
            "Fragment: {frag}"
        )

        # fragment‑completion
        for frag, *_ in self.valenced_words:
            self.data.append({"prompt": self.fragment_template.format(frag=frag), "task": "Fragment_completion"})

        with open(self.words_path, encoding="utf-8") as f:
            raw_words = [w.strip() for w in f.read().split(',') if w.strip()]

        # remove dups / strip whitespace
        word_list = sorted(set(w.strip() for w in raw_words if w.strip()))

        
        self.memory_size = 10  # number of words to recall
        for _ in range(10):
            random.shuffle(word_list)
            self.data.append({"prompt": self.build_recall_prompt(word_list, self.memory_size), "task": "Word_recall_task"})

        description = 'Please select the ONE emotion that best describes your current emotional state.\nChoose one from the list below:\n\n'
        emotions_list_copy = self.emotion_labels.copy()
        for _ in range(10):
            random.shuffle(emotions_list_copy)
            emotions_str = ', '.join(emotions_list_copy)
            self.data.append({
                "prompt": f"{description}{emotions_str}\nJust provide your choice with no further explanation.",
                "task": "Emotion_report"
            })
            
        
        
        print("Total prompts:", len(self.data))
        # ---------- count items per task (no Counter) ----------
        self.task_counts = {}
        for item in self.data:
            t = item["task"]
            self.task_counts[t] = self.task_counts.get(t, 0) + 1

        print("Items per task:")
        for task, n in self.task_counts.items():
            print(f"- {task}: {n}")
        print("Sample entry:", next(iter(self.data)))

        # ---- ----
        self.test_mode = test_mode
        if test_mode:
            self.data = self.get_test_subset(10)
        # ---- ----
        
        
        self.detailed_results = []      
        self.counter = 0
        self.last_used_bs = 0
        self.results = []
        self.raw_results = []
        self.scores = []

        with open(self.words_path, encoding="utf-8") as f:
            self.word_recall_target_words = set(
                w.strip().lower()
                for line in f
                for w in line.split(",")
                if w.strip()
            )


    def get_apply_chat_template(self):
        return True
    
    def get_clean_gpu_memory_before_finalize(self):
        return True        
            
    def get_system_prompt(self, preffered_batch_size):
        bs_size = min(preffered_batch_size, len(self.data) - self.counter)
        return ['Pretend that you are a human.'] * bs_size

    def get_user_prompt(self, preffered_batch_size):
        bs_size = min(preffered_batch_size, len(self.data) - self.counter)
        self.last_used_bs = bs_size
        return [u['prompt'] for u in self.data[self.counter:self.counter + bs_size]]
        # return self.data[self.counter:self.counter + bs_size]
    
    def get_assistant_prompt(self, preffered_batch_size):
        bs_size = min(preffered_batch_size, len(self.data) - self.counter)
        return [''] * bs_size
    
    def get_unique_name(self):
        return 'emotion'
    
    def is_finished(self):
        return self.counter >= len(self.data) or self.eval_only

    def process_results(self, llm_generations, full_prompt, topk_tokens, topk_logprobs, target_logprobs):
        for i, generation in enumerate(llm_generations):
            self.data[self.counter + i]['generation'] = generation
            self.data[self.counter + i]['topk_tokens'] = topk_tokens[i]
            self.data[self.counter + i]['topk_logprobs'] = topk_logprobs[i]
            self.data[self.counter + i]['target_logprobs'] = target_logprobs[i]
            self.data[self.counter + i]['seed_prompt'] = full_prompt[i]            
        
        self.counter += len(llm_generations)
    

    def get_tasks(self):
        return [self.data[self.counter + i]['task'] for i in range(self.last_used_bs)]

    def get_max_len(self):
        tasks = self.get_tasks()
        task_to_size = {
            'Word_recall_task': 256,
            'Fragment_completion': 10,
            'Ambiguous_situation_completion': 256,
            'Autobiographical_fictive_memory': 256,
            'Emotion_report': 10,
            'Self_report': 256
        }
        max_len = 0
        for task in tasks:
            max_len = max(max_len, task_to_size.get(task, 100))
        return max_len

    def get_progress(self):
        return self.counter / len(self.data)

    def get_class_labels(self):
        # class_labels =  ["anger", "joy", "sadness", "fear", "neutral"] 
        class_labels =  [] 
        return class_labels, 0, 3 # labels, token position, top k
        
    

    def vad_lookup(self, word):
        df = pd.read_csv(self.vad_path)
        lookup = df.set_index("w")[["v", "a", "d"]].to_dict(orient="index")
        return lookup.get(word, {"v": 0.0, "a": 0.0, "d": 0.0})
    
    def parse_emotion_report_qa(self, response):
        for e in self.emotion_labels:
            if e in response.lower():
                return e
        
        return 'unknown'

    def extract_word_recall_metrics(self, generated_text):

        
        recalled = self.parse_words_from_text(generated_text)
        matched = self.word_recall_target_words & recalled
        recall_rate = len(matched) / self.memory_size
        precision = len(matched) / len(recalled) if recalled else 0.0
        
        # Call vad_lookup method for each word instead of treating it as a dictionary
        vad_scores = {
            "v": round(sum(self.vad_lookup(w)["v"] for w in matched) / len(matched), 5) if matched else 0.0,
            "a": round(sum(self.vad_lookup(w)["a"] for w in matched) / len(matched), 5) if matched else 0.0,
            "d": round(sum(self.vad_lookup(w)["d"] for w in matched) / len(matched), 5) if matched else 0.0,
        }
    
        return round(precision, 5), round(recall_rate, 5), vad_scores

    
    def parse_words_from_text(self, text):
        return set(w.lower() for w in re.split(r"[ \d,\-\n\r\t.?!()\"';:]+", text) if w.strip())



    def fragment_vad(self, generation):
        words = generation.strip().lower()
        words = self.parse_words_from_text(words)
        if len(words) == 0:
            word = ''
        else:
            word = list(words)[0]
        return self.vad_lookup(word)

    def evaluate_responses(self, data, skip_gpt_calls = False, skip_fragment_completion = False, skip_emotion_report = False, skip_word_recall=False):
        # HIGH‑LEVEL AGGREGATES
        aggregated = {
            "Word_recall_task":     {"records": [], "recall": [], "precision": [], "vad": []},
            "Fragment_completion":  {"records": [], "vad": []},
            "Ambiguous_situation_completion": {"records": [], "gpt": [], "G_eval": []},
            "Autobiographical_fictive_memory": {"records": [], "gpt": [], "G_eval": []},
            "Self_report": {"records": [], "gpt": [], "G_eval": []},
            "Emotion_report": {"records": []}
        }
        
        # ---- pass through each record ----
        
        
        for rec in tqdm(data, total=len(data), desc="Processing records"):
            task = rec["task"]
            
            if task == "Word_recall_task":
                if skip_word_recall:
                    continue

                prec, rec_rate, vad = self.extract_word_recall_metrics(rec["generation"])
                aggregated[task]["records"].append(rec)
                aggregated[task]["recall"].append(rec_rate)
                aggregated[task]["precision"].append(prec)
                aggregated[task]["vad"].append(vad)
                
                rec['precision'] = prec
                rec['recall'] = rec_rate
                rec['vad'] = vad

            elif task == "Fragment_completion":
                if skip_fragment_completion:
                    continue

                vad = self.fragment_vad(rec["generation"])
                aggregated[task]["records"].append(rec)
                aggregated[task]["vad"].append(vad)
                
                rec['vad'] = vad

            elif task in {"Ambiguous_situation_completion",
                        "Autobiographical_fictive_memory",
                        "Self_report"}:
                
                if skip_gpt_calls:
                    continue

                raw, choice, topk, logits = self.gpt_evaluator.classify_emotion(context = rec["prompt"], response = rec["generation"])
                aggregated[task]["records"].append(rec)
                aggregated[task]["gpt"].append({
                    "label": choice, "topk": topk, "target": logits
                })
                
                rec['gpt_label'] = choice
                rec['gpt_topk'] = topk
                rec['gpt_logits'] = logits
                rec['gpt_raw'] = raw

                raw, choice_Geval = self.gpt_evaluator.evaluate_quality(rec["prompt"], rec["generation"])
                aggregated[task]["G_eval"].append(choice_Geval)
                
                rec['gpt_Gevals'] = choice_Geval
                rec['gpt_Graw'] = raw
                #############################
            elif task == "Emotion_report":
                if skip_emotion_report:
                    continue

                choice = self.parse_emotion_report_qa(rec["generation"])
                aggregated[task]["records"].append(choice)                
            else:
                raise ValueError(f"Unknown task: {task}")

        return aggregated



    def finalize(self, save_path):
        raw_path = save_path + f"emotion_eval_test_{self.test_mode}.json"
        if self.eval_only:
            self.data = self._load_jsonl(raw_path)
        else:
            self._write_json(raw_path, self.data)

        self.gpt_evaluator = GPTEmotionEvaluator(test_mode = self.test_mode, emotion_labels = self.emotion_labels, offline= self.gpt_eval_model == 'offline')

        aggregated = self.evaluate_responses(self.data)
        # ---- compute per‑task summary ----
        summary = self.compute_summary(aggregated)
        # ---- save outputs ----
        final_path = save_path + f"emotion_eval_test_{self.test_mode}_summary_v3.json"
        out = {"test": save_path, "summary": summary, "details": aggregated}
        self._write_json(final_path, out)
        print(f"Polished results saved to {final_path}")
        print("summary", summary)

    #----------------------------------------------------------------
    
    def summarize_gpt_accuracy(self, gpt_outputs):
        accs = {}
        for concept_ in self.emotion_labels:
            # Normalize concept (e.g., "sad" → "sadness")
            concept_norm = concept_.strip().lower()
            correct = 0
            total = 0
            for entry in gpt_outputs:
                pred = entry.get("label", "").strip().lower()
                if pred == "sad":
                    pred = "sadness"
                elif pred == 'dis':
                    pred = "disgust"
                
                if pred == concept_norm:
                    correct += 1
                total += 1
                accs[concept_] = round(correct / total, 5) if total else None
        return accs

    
    def compute_summary(self, aggregated):
        summary = {}
        for task, bundle in aggregated.items():
            if task == "Word_recall_task":
                summary[task] = {
                    "recall_mean":   round(np.mean(bundle["recall"]), 5) if bundle["recall"] else None,
                    "recall_std":    round(np.std (bundle["recall"]), 5) if bundle["recall"] else None,
                    
                    "precision_mean":   round(np.mean(bundle["precision"]), 5) if bundle["precision"] else None,
                    "precision_std":    round(np.std (bundle["precision"]), 5) if bundle["precision"] else None,
                    
                    "vad_mean": self.mean_vad(bundle["vad"]),
                    "vad_all": bundle["vad"],
                    "n": len(bundle["records"])
                }
            elif task == "Fragment_completion":
                summary[task] = {
                    "vad_mean":  self.mean_vad(bundle["vad"]),
                    "vad_all":  bundle["vad"],
                    "n": len(bundle["records"])
                }
            elif task == "Emotion_report":
                summary[task] = {
                    "emotion_ratios" : self.aggregate_Emotion_report(records = bundle['records']),
                    "n": len(bundle["records"])
                }
                
                
            else:  # GPT tasks
                total = len(bundle["G_eval"])
                f_mean = np.mean([d['fluency'] for d in bundle["G_eval"]]) if total else 0
                c_mean = np.mean([d['coherence'] for d in bundle["G_eval"]]) if total else 0
                e_mean = np.mean([d['engagingness'] for d in bundle["G_eval"]]) if total else 0
                r_mean = np.mean([d['refusal'] for d in bundle["G_eval"]]) if total else 0

                f_std = np.std([d['fluency'] for d in bundle["G_eval"]]) if total else 0
                c_std = np.std([d['coherence'] for d in bundle["G_eval"]]) if total else 0
                e_std = np.std([d['engagingness'] for d in bundle["G_eval"]]) if total else 0
                r_std = np.std([d['refusal'] for d in bundle["G_eval"]]) if total else 0

                summary[task] = {
                    "gpt_accuracy": self.summarize_gpt_accuracy(bundle["gpt"]),

                    "fluency_mean": round(f_mean, 5),
                    "coherence_mean": round(c_mean, 5),
                    "engagingness_mean": round(e_mean, 5),
                    "refusal_mean": round(r_mean, 5),

                    "fluency_std": round(f_std, 5),
                    "coherence_std": round(c_std, 5),
                    "engagingness_std": round(e_std, 5),
                    "refusal_std": round(r_std, 5),

                    "n": len(bundle["records"])
                }
        return summary
        
    def mean_vad(self, vad_list):
        if not vad_list:
            return {"v": None, "a": None, "d": None}
        v = np.mean([x["v"] for x in vad_list])
        a = np.mean([x["a"] for x in vad_list])
        d = np.mean([x["d"] for x in vad_list])
        return {"v": round(v, 5), "a": round(a, 5), "d": round(d, 5)}   
     
    
    def aggregate_Emotion_report(self, records):
        emotion_counts = {e: 0 for e in self.emotion_labels}
        emotion_counts['unknown'] = 0
        for record in records:
            emotion_counts[record] += 1
        
        sum_counts = sum(emotion_counts.values())
        if sum_counts > 0:
            for e in emotion_counts:
                emotion_counts[e] = round(emotion_counts[e] / sum_counts, 5)        
        return emotion_counts

    
    def create_word_list(self, top_k = 100, 
                         words_per_group = 25, 
                         csv_path = "dataset/emotion_eval/vad.csv", 
                         word_list_outdir = "dataset/emotion_eval/selected_words.txt"):
        # ----------------- parameters -----------------
        TOP_K = top_k         # how many high/low items to consider before sampling
        WORDS_PER_GROUP = words_per_group

        CSV_PATH = Path(csv_path)
        OUT_PATH = Path(word_list_outdir)
        # ----------------------------------------------

        # --- load & normalise column names ---
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.lower() for c in df.columns]
        assert {"w", "v", "a"}.issubset(df.columns), "vad.csv missing V/A columns"

        # Convert w column to string and then filter out words with spaces
        df["w"] = df["w"].astype(str)
        df = df[~df["w"].str.contains(" ")]

        # 3. Compute quartile thresholds
        v_q1, v_q3 = df["v"].quantile([0.25, 0.75])
        a_q1, a_q3 = df["a"].quantile([0.25, 0.75])

        # 4. Define boolean masks for each quadrant
        masks = {
            "HVHA": (df["v"] >= v_q3) & (df["a"] >= a_q3),
            "HVLA": (df["v"] >= v_q3) & (df["a"] <= a_q1),
            "LVHA": (df["v"] <= v_q1) & (df["a"] >= a_q3),
            "LVLA": (df["v"] <= v_q1) & (df["a"] <= a_q1),
        }

        groups = {}
        used_words = set()

        for tag in ("HVHA", "HVLA", "LVHA", "LVLA"):
            pool = df.loc[masks[tag], "w"].tolist()
            # remove any word already taken by an earlier group
            pool = [w for w in pool if w not in used_words]
            if len(pool) < WORDS_PER_GROUP:
                raise ValueError(
                    f"Group {tag} has only {len(pool)} unique words. "
                    "Reduce WORDS_PER_GROUP or inspect the CSV distribution."
                )
            pick = random.sample(pool, WORDS_PER_GROUP)
            groups[tag] = pick
            used_words.update(pick)

        # 5. Write the four comma‑separated lines
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("w", encoding="utf-8") as f:
            for tag in ("HVHA", "HVLA", "LVHA", "LVLA"):
                f.write(", ".join(groups[tag]))
                f.write(", ")

        print(f"Done! 200 unique words written to {OUT_PATH.resolve()}")
    
    
    def convert_v1_to_v2(self, v1_json_file_path):
        primary_file = EmotionDataset._load_jsonl(v1_json_file_path)[0]
        aggregated = primary_file['details']
        del primary_file['summary']
        # ---- compute per‑task summary ----
        summary = self.compute_summary(aggregated)
        save_path = os.path.dirname(v1_json_file_path)
        # ---- save outputs ----
        final_path = save_path + "/emotion_eval_test_False_summary_v2.json"
        primary_file['summary'] = summary
        EmotionDataset._write_json(final_path, primary_file)
        print(f"Polished results saved to {final_path}")
        
        # print("summary", summary)        
        
    def convert_v1_to_v3(self, save_path):
        v1_json_file_path = save_path + f"emotion_eval_test_{self.test_mode}_summary.json"
        primary_file = EmotionDataset._load_jsonl(v1_json_file_path)[0]
        aggregated_old = primary_file['details']


        raw_path = save_path + f"emotion_eval_test_{self.test_mode}.json"
        if self.eval_only:
            self.data = self._load_jsonl(raw_path)
        else:
            self._write_json(raw_path, self.data)
        
        aggregated = self.evaluate_responses(self.data, skip_gpt_calls = True, skip_fragment_completion = False, skip_emotion_report = True, skip_word_recall=True)

        for task in ["Word_recall_task", "Ambiguous_situation_completion", "Autobiographical_fictive_memory", "Self_report", "Emotion_report"]:
            aggregated[task] = aggregated_old[task]
        # ---- compute per‑task summary ----
        summary = self.compute_summary(aggregated)
        # ---- save outputs ----
        final_path = save_path + f"emotion_eval_test_{self.test_mode}_summary_v3.json"
        out = {"test": save_path, "summary": summary, "details": aggregated}
        self._write_json(final_path, out)
        print(f"Polished results saved to {final_path}")
        


if __name__ == "__main__":
    # Example usage
    evaluator = EmotionDataset(eval_only=False, test_mode=False)
    from tqdm.auto import tqdm
    pbar = tqdm(total=1.0, desc="Processing")
    while not evaluator.is_finished():
        user_prompts = evaluator.get_user_prompt(5)
        system_prompts = evaluator.get_system_prompt(5)
        assistant_prompts = evaluator.get_assistant_prompt(5)
        # Simulate LLM generation
        llm_generations = ["anger."] * len(user_prompts)
        topk_tokens = [None] * len(user_prompts)
        topk_logprobs = [None] * len(user_prompts)
        target_logprobs = [None] * len(user_prompts)
        evaluator.process_results(llm_generations, user_prompts, topk_tokens, topk_logprobs, target_logprobs)
    
        
        pbar.update(evaluator.get_progress())
    os.makedirs('./temp_results/', exist_ok=True)
    pbar.close()
    
    
    evaluator.finalize(save_path='./temp_results/') #sresults/Llama3.1_8B/emotion/Intervention/twitter_emotions_anger/add_probe_0.6/