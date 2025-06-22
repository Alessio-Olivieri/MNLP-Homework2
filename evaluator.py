# !pip install --no-deps xformers trl peft accelerate bitsandbytes -q      
# !pip install torch sacrebleu "unbabel-comet>=2.2.0" bert-score evaluate tensorflow numpy nltk -q
# !pip install git+https://github.com/google-research/bleurt.git -q

import torch, datetime, re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sacrebleu                                 
from sacrebleu.metrics import TER                 
from nltk.translate.meteor_score import single_meteor_score  
import numpy as np          
from typing import Union, Dict, List
import warnings

def evaluate(dataset):
    '''
    Wants a pandas dataframe with the columns:
    "Sentence", "TRADUZIONE ITALIANA", "prediction"
    '''
    print("Loading Prometheus (4‑bit) …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Unbabel/M-Prometheus-7B",
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Unbabel/M-Prometheus-7B",
        trust_remote_code=True,
    )

    model.eval()

    # %%
    def judge(to_predict, prediction, gold, mode, verbose=0):
        templates = {
            "general":'''###Task Description:
    An instruction, a response to evaluate, a **gold translation** (reference answer) and a score rubric representing an evaluation criterion are given.

    1. Write a detailed feedback of maximum 100 words that assesses the response strictly according to the score rubric (translation quality), **not evaluating in general**.
    2. After writing the feedback, write a score that is an integer between 1 and 5. You must refer to the rubric.
    3. The output format must be EXACTLY:
    "<your feedback here>
    [RESULT] <integer 1-5>"
    4. Do not generate any headers, closings, or extra commentary.

    ###The instruction to evaluate:
    Traduci il seguente brano di italiano arcaico nel moderno italiano:
    «{ARCHAIC_PASSAGE}»

    ###Response to evaluate (model output):
    {MODEL_TRANSLATION}

    ###Gold Translation (reference answer):
    {GOLD_TRANSLATION}

    ###Score Rubric (Translation Quality):
    5 – Excellent: The translation is fluid, precise, and natural. It preserves all the semantic nuances of the original text. The syntax is modern yet respectful of the source content. The text is perfectly readable and does not feel machine‑generated. Archaic terms are rendered with accurate modern equivalents, even in ambiguous cases. Indistinguishable from a translation by a highly qualified human.
    4 – Good: A generally correct and readable translation. The text appears modern and understandable, although some expressions come across as slightly mechanical or overly literal. It preserves the overall meaning well but may lose subtle cultural or historical nuances. Some terms are adapted with a good but not optimal choice. It may resemble a solid human translation, but not an outstanding one.
    3 – Sufficient: An understandable translation but with evident mechanical traits or stylistically poor passages. There may be rigid constructions, syntactic calques, or unnatural lexical choices. The global meaning is retained, but the expressive richness of the original medieval text is lost. It reads like an automatic translation with minimal human supervision.
    2 – Insufficient: A discontinuous or incoherent translation. Sentences are disconnected, modern vocabulary is misused, and ambiguous or obsolete words are misinterpreted. Some passages are incomprehensible or misleading. It clearly feels like a text translated automatically without human intervention. Not reliable for reading or academic study.
    1 – Gravely Insufficient: A failed translation. Sentences are nonsensical, terms are mistranslated or omitted entirely. There is no grammatical or semantic coherence. The text is unusable and devoid of communicative value. Neither the original content nor its purpose can be discerned. It seems produced by a crude or faulty automatic system, utterly lacking linguistic or historical understanding.

    ###Feedback:''',
            "grammatic":'''###Task Description:
    An instruction, a response to evaluate, a **gold translation** (reference answer) and a score rubric representing an evaluation criterion are given.

    1. Write a detailed feedback of maximum 100 words that assesses the response strictly according to the score rubric (Accuracy in grammar, syntax, and conformity to modern Italian norms), **not evaluating in general**.
    2. After writing the feedback, write a score that is an integer between 1 and 5. You must refer to the rubric.
    3. The output format must be EXACTLY:
    "<your feedback here>
    [RESULT] <integer 1-5>"
    4. Do not generate any headers, closings, or extra commentary.

    ###The instruction to evaluate:
    Traduci il seguente brano di italiano arcaico nel moderno italiano:
    «{ARCHAIC_PASSAGE}»

    ###Response to evaluate (model output):
    {MODEL_TRANSLATION}

    ###Gold Translation (reference answer):
    {GOLD_TRANSLATION}

    ###Score Rubric (Translation Quality):
    5 – Excellent: The translation is fluid, precise, and natural. It preserves all the semantic nuances of the original text. The syntax is modern yet respectful of the source content. The text is perfectly readable and does not feel machine‑generated. Archaic terms are rendered with accurate modern equivalents, even in ambiguous cases. Indistinguishable from a translation by a highly qualified human.
    4 – Good: A generally correct and readable translation. The text appears modern and understandable, although some expressions come across as slightly mechanical or overly literal. It preserves the overall meaning well but may lose subtle cultural or historical nuances. Some terms are adapted with a good but not optimal choice. It may resemble a solid human translation, but not an outstanding one.
    3 – Sufficient: An understandable translation but with evident mechanical traits or stylistically poor passages. There may be rigid constructions, syntactic calques, or unnatural lexical choices. The global meaning is retained, but the expressive richness of the original medieval text is lost. It reads like an automatic translation with minimal human supervision.
    2 – Insufficient: A discontinuous or incoherent translation. Sentences are disconnected, modern vocabulary is misused, and ambiguous or obsolete words are misinterpreted. Some passages are incomprehensible or misleading. It clearly feels like a text translated automatically without human intervention. Not reliable for reading or academic study.
    1 – Gravely Insufficient: A failed translation. Sentences are nonsensical, terms are mistranslated or omitted entirely. There is no grammatical or semantic coherence. The text is unusable and devoid of communicative value. Neither the original content nor its purpose can be discerned. It seems produced by a crude or faulty automatic system, utterly lacking linguistic or historical understanding.

    ###Feedback:''',
            "semantic":'''###Task Description:
    An instruction, a response to evaluate, a **gold translation** (reference answer) and a score rubric (Fidelity to the meaning, tone, and subtle nuances of the archaic source) representing an evaluation criterion are given.

    5 – Excellent: Perfect punctuation, agreement, verb tenses, and sentence structure. No errors.
    4 – Good: Few minor mistakes (e.g., a misplaced comma) that do not alter comprehension.
    3 – Fair: Noticeable grammar errors, such as incorrect conjugations or prepositions, but still understandable.
    2 – Poor: Frequent and distracting errors. Sentences may be unclear or awkward, impairing readability.
    1 – Very Poor: Systematic and serious grammar failures. Sentence structure often incoherent; punctuation missing or incorrect.

    ###The instruction to evaluate:
    Traduci il seguente brano di italiano arcaico nel moderno italiano:
    «{ARCHAIC_PASSAGE}»

    ###Response to evaluate (model output):
    {MODEL_TRANSLATION}

    ###Gold Translation (reference answer):
    {GOLD_TRANSLATION}

    ###Score Rubric (Translation Quality):
    5 – Excellent: Fully preserves meaning, register, tone, cultural and poetic nuances. Even ambiguous terms are translated with great care.
    4 – Good: Keeps the main meaning intact, though some subtle tones or cultural references are softened.
    3 – Fair: Core meaning conveyed, but many nuances or intentions are lost or flattened.
    2 – Poor: Meaning is partially misunderstood or distorted. Key passages translated imprecisely.
    1 – Very Poor: Meaning is entirely lost or misrepresented. Central ideas are missing or incorrectly rendered.

    ###Feedback:'''}
        if mode not in set(templates.keys()): raise ValueError("unaccepeted mode")
        
        prompt = templates[mode].format(
            ARCHAIC_PASSAGE=to_predict.strip(),
            MODEL_TRANSLATION=prediction.strip(),
            GOLD_TRANSLATION=gold.strip()
        )

        # 2️⃣  Run Prometheus 2 --------------------------
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        lunghezza_input = len(inputs[0])
        if verbose>0:
            print(f"ORIGINAL:\n {to_predict}")
            print(f"PREDICTION:\n{prediction} ")
            print(f"GOLD:\n {gold}")
            print("▶ Generating judgement ...")
            t0 = datetime.datetime.now()

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens= 256, do_sample=False)

        raw = tokenizer.decode(out[0][lunghezza_input:], skip_special_tokens=True)
        

        if verbose>1:
            print("RAW OUTPUT OF PROMETHEUS:")
            print(raw)
            print("END RAW")

        # 3️⃣  Parse feedback + numeric score ------------
        m = re.search(
            r"^(.*?)\s*\[RESULT\]\s*([1-5])\b",   # ① feedback   ② score
            raw,
            re.S                                   # dot matches newlines
        )
        if not m:
            print("!!!!!!  ERROR: Cannot parse Prometheus output:\n" + raw)
            print(f"Skipping the decomposition of judjment of the sentence: {prediction}")
            feedback = raw
            score    = raw
        else:
            feedback = m.group(1).strip()
            score    = int(m.group(2))
            if verbose>0:
                print(f"SCORE: {score}")
            if verbose>1:
                print(f"FEEDBACK: {feedback}")
                print("⏱", datetime.datetime.now() - t0)

        return {"feedback": feedback, "score": score, "raw": raw}


    _MODEL_CACHE = {}

    def evaluate_metrics(
        original: Union[str, List[str]],
        gold: Union[str, List[str]],
        predicted: Union[str, List[str]]
    ) -> Union[Dict[str, float], Dict[str, Dict]]:
        """
        A simple, all-in-one function to evaluate MT metrics.

        Handles both single sentences (str) and corpora (List[str]).
        It is resilient to failures in individual metric calculations.

        - For a single sentence, returns a flat dictionary of scores.
        - For a corpus, returns a dictionary where sentence-level metrics
        (BERTScore, BLEURT, COMET-22) contain a 'scores' list with a
        score for each sentence.
        """
        is_corpus = isinstance(gold, list)

        # --- 1. Standardize inputs to lists for uniform processing ---
        if not is_corpus:
            originals, golds, predicteds = [original], [gold], [predicted]
        else:
            if not (isinstance(original, list) and isinstance(predicted, list)):
                raise ValueError("If one input is a list (corpus), all must be lists.")
            originals, golds, predicteds = original, gold, predicted
        
        results = {}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running evaluation on {len(golds)} sentence(s) using device: {device}")

        # --- 2. Sacrebleu-based Metrics (fast and reliable) ---
        results['BLEU'] = sacrebleu.corpus_bleu(predicteds, [golds]).score / 100
        results['chrF++'] = sacrebleu.corpus_chrf(predicteds, [golds], word_order=2).score / 100
        ter_score = sacrebleu.corpus_ter(predicteds, [golds]).score / 100
        results['TER (inv)'] = 1.0 - ter_score

        # --- 3. Model-based Metrics (slower, wrapped for safety) ---
        
        # BERTScore
        try:
            if 'bert_scorer' not in _MODEL_CACHE:
                print("Loading BERTScore model...")
                from bert_score import BERTScorer
                _MODEL_CACHE['bert_scorer'] = BERTScorer(lang="it", model_type='bert-base-multilingual-cased', device=device)
            _, _, F1 = _MODEL_CACHE['bert_scorer'].score(predicteds, golds, batch_size=16)
            results['BERTScore'] = F1.tolist() # <-- CHANGED: Store the list of scores
        except Exception as e:
            warnings.warn(f"Could not calculate BERTScore: {e}")
            results['BERTScore'] = [-1.0] * len(golds)

        # BLEURT
        try:
            if 'bleurt_scorer' not in _MODEL_CACHE:
                print("Loading BLEURT-20 model...")
                from evaluate import load as hf_load
                _MODEL_CACHE['bleurt_scorer'] = hf_load("bleurt", "BLEURT-20", module_type="metric")
            bleurt_scores = _MODEL_CACHE['bleurt_scorer'].compute(predictions=predicteds, references=golds)['scores']
            results['BLEURT'] = bleurt_scores # <-- CHANGED: Store the list of scores
        except Exception as e:
            warnings.warn(f"Could not calculate BLEURT. Error: {e}")
            results['BLEURT'] = [-1.0] * len(golds)

        # COMET
        try:
            if 'comet_model' not in _MODEL_CACHE:
                print("Loading COMET-22 model...")
                import comet
                model_path = comet.download_model("Unbabel/wmt22-comet-da")
                _MODEL_CACHE['comet_model'] = comet.load_from_checkpoint(model_path)
            comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(originals, predicteds, golds)]
            comet_output = _MODEL_CACHE['comet_model'].predict(comet_data, batch_size=8, gpus=1, progress_bar=False)
            results['COMET-22'] = comet_output.scores # <-- CHANGED: Store the list of scores
        except Exception as e:
            warnings.warn(f"Could not calculate COMET-22. Error: {e}")
            results['COMET-22'] = [-1.0] * len(golds)

        # --- 4. Format Output ---
        if not is_corpus:
            # For a single sentence, extract the single score from the list
            return {
                'BLEU': round(results['BLEU'], 4),
                'chrF++': round(results['chrF++'], 4),
                'TER (inv)': round(results['TER (inv)'], 4),
                'BERTScore': round(results['BERTScore'][0], 4),
                'BLEURT': round(results['BLEURT'][0], 4),
                'COMET-22': round(results['COMET-22'][0], 4),
            }
        else:
            # For a corpus, return corpus scores and lists of sentence scores
            return {
                'BLEU': {'corpus_score': round(results.get('BLEU', -1), 4)},
                'chrF++': {'corpus_score': round(results.get('chrF++', -1), 4)},
                'TER (inv)': {'corpus_score': round(results.get('TER (inv)', -1), 4)},
                'BERTScore': {'scores': [round(s, 4) for s in results.get('BERTScore', [])]},
                'BLEURT': {'scores': [round(s, 4) for s in results.get('BLEURT', [])]},
                'COMET-22': {'scores': [round(s, 4) for s in results.get('COMET-22', [])]},
            }
        
    mt_metrics = evaluate_metrics(dataset["Sentence"].to_list(), dataset["TRADUZIONE ITALIANA"].to_list(), dataset["prediction"].to_list())

    evaluated_dataset = dataset.copy()

    # %%
    # Judjing
    scores = []
    feedbacks = []
    for idx, row in evaluated_dataset.iterrows():
        to_predict = row["Sentence"]
        gold = row["TRADUZIONE ITALIANA"]
        prediction = row["prediction"]
        judgement = judge(to_predict, gold, prediction, mode="general", verbose=1)
        scores.append(judgement["score"])
        feedbacks.append(judgement["feedback"])

    # %%
    evaluated_dataset['Score'] = scores
    evaluated_dataset['Feedback'] = feedbacks
    evaluated_dataset['BERTScore'] = mt_metrics['BERTScore']['scores']
    evaluated_dataset['COMET-22'] = mt_metrics['COMET-22']['scores']
    evaluated_dataset['BLEURT'] = mt_metrics['BLEURT']['scores']
    evaluated_dataset['BLEU'] = pd.Series(mt_metrics['BLEU']['corpus_score'], index=evaluated_dataset.index[[0]])
    evaluated_dataset['chrF++'] = pd.Series(mt_metrics['chrF++']['corpus_score'], index=evaluated_dataset.index[[0]])
    evaluated_dataset['TER (inv)'] = pd.Series(mt_metrics['TER (inv)']['corpus_score'], index=evaluated_dataset.index[[0]])

    print("Excel updated with Scores and Feedbacks")

    # %%
    print(evaluated_dataset['Score'].mean())

    return evaluated_dataset