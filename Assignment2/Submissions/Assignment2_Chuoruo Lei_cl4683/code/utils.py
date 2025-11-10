import sys, types, importlib.machinery

#  屏蔽系统残留 bitsandbytes
fake_bnb = types.ModuleType("bitsandbytes")
fake_bnb.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)
fake_bnb.nn = None
sys.modules["bitsandbytes"] = fake_bnb


import re
import os
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer

lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    'llama3.1': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    'deepseek': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
}

def tokenize(args, tokenizer, feature):
    """
    Tokenize prompt + answer, truncate instead of dropping overlength samples.
    """
    # Prompt 和 Answer 各取一半长度，防止超长被删光
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(),
        padding=False,
        max_length=args.max_length // 2,
        truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(),
        padding=False,
        max_length=args.max_length // 2,
        truncation=True,
        add_special_tokens=False
    )
    
    # 拼接 + 截断，不删除
    input_ids = (prompt_ids + target_ids)[: args.max_length - 1]
    input_ids.append(tokenizer.eos_token_id)

    # ❗️关键：prompt 部分的 labels 置为 -100，避免纳入 loss
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    labels = labels[: len(input_ids)]  # 防御性截断

    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceed_max_length": False,  # 我们已改为截断，不再丢样本
    }

def parse_answer(text):
    if not text or not isinstance(text, str):
        return {
            "positive developments": None,
            "potential concerns": None,
            "prediction": 0.0,
            "prediction_binary": 0,
            "analysis": "unparsed"
        }

    clean_text = text.strip()

    result = {
        "positive developments": None,
        "potential concerns": None,
        "prediction": None,
        "prediction_binary": None,
        "analysis": None
    }

    # === 1️⃣ 标签匹配 ===
    pos_match = re.search(
        r"(?is)(?:Positive[s]?|Positive Factors?|Positive Developments?)[:：]\s*(.+?)(?=(?:Negative|Potential|Concern|Forecast|Prediction|Analysis|Sentiment|Outlook|$))",
        clean_text)
    neg_match = re.search(
        r"(?is)(?:Negative[s]?|Negative Factors?|Potential Concerns?|Concerns?)[:：]\s*(.+?)(?=(?:Forecast|Prediction|Analysis|Sentiment|Outlook|$))",
        clean_text)
    pred_match = re.search(
        r"(?is)(?:Prediction|Forecast|Analysis|Outlook|Sentiment)[:：→]?\s*(.+)",
        clean_text)

    if pos_match:
        result["positive developments"] = pos_match.group(1).strip()
    if neg_match:
        result["potential concerns"] = neg_match.group(1).strip()
    if pred_match:
        result["analysis"] = pred_match.group(1).strip()

    # === 2️⃣ 启发式补充 ===
    if not result["positive developments"]:
        guess = re.search(
            r"((?:Strong|Improving|Upside|Growth|Profit|Revenue|Buyback|Repurchase|Beat|Optimized?|Earnings beat)[^.]+)",
            clean_text, re.I)
        if guess:
            result["positive developments"] = guess.group(1).strip()

    if not result["potential concerns"]:
        guess = re.search(
            r"((?:However|But|Downside|Risk|Concern|Debt|Weak|Pressure|Cost|Margin)[^.]+)",
            clean_text, re.I)
        if guess:
            result["potential concerns"] = guess.group(1).strip()

    if not result["analysis"]:
        guess = re.search(
            r"((?:Expect|Forecast|Overall|Likely|Bullish|Bearish|Gain|Decline|Upside|Downside|Sentiment)[^.]+)",
            clean_text, re.I)
        if guess:
            result["analysis"] = guess.group(1).strip()

    # === 3️⃣ 若全空 ===
    if not any([result["positive developments"], result["potential concerns"], result["analysis"]]):
        result["analysis"] = "unparsed"
        result["prediction"] = 0.0
        result["prediction_binary"] = 0
        return result

    # === 4️⃣ “but / however” 拆句逻辑 ===
    if result["positive developments"] and re.search(r"\bbut\b|\bhowever\b", result["positive developments"], re.I):
        parts = re.split(r"\bbut\b|\bhowever\b", result["positive developments"], flags=re.I)
        if len(parts) > 1:
            result["potential concerns"] = parts[-1].strip()
            result["positive developments"] = parts[0].strip(" ,.")

    # === 5️⃣ 方向判定 ===
    if result["analysis"]:
        ana = result["analysis"]
        perc = re.search(r"([-+]?\d+(\.\d+)?)\s*%", ana)
        if perc:
            result["prediction"] = float(perc.group(1))
        else:
            has_pos = re.search(r"(rise|increase|gain|bullish|up|improve|growth|positive|modest|upside|beat|buyback|optimistic)", ana, re.I)
            has_neg = re.search(r"(risk|drop|decrease|fall|bearish|down|decline|negative|loss|pressure|cost|margin)", ana, re.I)
            has_flat = re.search(r"(flat|stable|neutral|sideways|limited upside)", ana, re.I)

            if has_pos and not has_neg:
                result["prediction"] = 1.0
            elif has_neg and not has_pos:
                result["prediction"] = -1.0
            elif has_pos and has_neg:
                result["prediction"] = 1.0
            elif has_flat:
                result["prediction"] = 0.0
            else:
                result["prediction"] = 0.0

        result["prediction_binary"] = 1 if result["prediction"] > 0 else 0

    # === 6️⃣ 清理 ===
    for k, v in result.items():
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
            if ". " in v:
                v = v.split(". ")[0] + "."
            result[k] = v if v else None

    if result["positive developments"] and result["potential concerns"]:
        if result["potential concerns"] in result["positive developments"]:
            result["potential concerns"] = None

    if not any(result.values()):
        result["analysis"] = "unparsed"
        result["prediction"] = 0.0
        result["prediction_binary"] = 0

    return result


def calc_rouge_score(references, answers):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 过滤 None 或空字符串
    valid_pairs = [(ref, ans) for ref, ans in zip(references, answers)
                   if isinstance(ref, str) and isinstance(ans, str)
                   and ref.strip() and ans.strip()]
    
    if not valid_pairs:  # 防止空列表出错
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in valid_pairs]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}


    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
    