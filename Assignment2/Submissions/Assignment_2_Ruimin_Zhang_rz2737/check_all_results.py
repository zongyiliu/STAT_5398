from collections import defaultdict
import pickle
import pandas as pd

from utils import calc_rouge_score, parse_answer


llama_base = pd.read_pickle("comparison_results/llama3_base_metrics.pkl")
print("Llama3 Base Model Metrics:", len(llama_base))
llama_ft = pd.read_pickle("comparison_results/llama3_fine_tuned_metrics.pkl")
print("Llama3 ft Metrics:", len(llama_ft))
deepseek_base = pd.read_pickle("comparison_results/deepseek_base_metrics.pkl")
print("DeepSeek Base Model Metrics:", len(deepseek_base))  
deepseek_ft = pd.read_pickle("comparison_results/deepseek_fine_tuned_metrics.pkl")
print("DeepSeek ft Metrics:", len(deepseek_ft))


llama2_base = pd.read_pickle("comparison_results/llama2_base_metrics.pkl")
print("Llama2 Base Model Metrics:", len(llama2_base))
llama2_base_ans = pd.read_pickle("comparison_results/llama2_base_answers.pkl")
print("Llama2 Base Model Answers:", len(llama2_base_ans))
llama2_ft = pd.read_pickle("comparison_results/llama2_finetuned_metrics.pkl")
print("Llama2 ft Metrics:", len(llama2_ft))
llama2_ft_ans = pd.read_pickle("comparison_results/llama2_finetuned_answers.pkl")
print("Llama2 ft Answers:", len(llama2_ft_ans))

deepseek_base_ans = pd.read_pickle("comparison_results/deepseek_base_answers.pkl")
print("DeepSeek Base Model Answers:", len(deepseek_base_ans))
deepseek_ft_ans = pd.read_pickle("comparison_results/deepseek_finetuned_answers.pkl")
print("DeepSeek ft Answers:", len(deepseek_ft_ans))


import re
import pickle
import pandas as pd
from collections import defaultdict
from utils import parse_answer, calc_rouge_score  # 用你项目里的

import re
from collections import defaultdict
import pandas as pd
import pickle

# 保留 utils 里的 ROUGE 计算
from utils import calc_rouge_score
# 不再 from utils import parse_answer —— 我们在本文件里重写一个更鲁棒的
# from utils import parse_answer

def _clean_text(s: str) -> str:
    if not s:
        return ""
    # 去掉模型特殊 token 或 [INST] 包裹
    s = re.sub(r"^<[^>]+>", "", s).strip()
    m = re.search(r"\[/INST\]\s*(.*)", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s

def _normalize_direction(line: str):
    """把 'Up by 2-3%' / 'Likely to rise' 等归一到 up/down/flat。"""
    t = (line or "").lower()
    if re.search(r"\b(up|rise|rising|increase|gains?|bullish|go up)\b", t):
        return "up"
    if re.search(r"\b(down|fall|falling|decrease|drop|decline|bearish|go down)\b", t):
        return "down"
    if re.search(r"\b(flat|unchanged|neutral|sideways|stable|range[- ]?bound)\b", t):
        return "flat"
    return None

def parse_answer_relaxed(ans: str):
    """
    兼容你当前模型输出：
    [Positive Developments]、[Potential Concerns]、[Prediction & Analysis]
    以及 'Prediction: Up by 2-3%' 这种格式。
    返回字典：{'positive developments': str, 'potential concerns': str, 'analysis': str, 'prediction': 'up|down|flat|None'}
    """
    text = _clean_text(ans)

    def _grab(tag_from, tag_to_list):
        # 截取 [tag_from] 到后面某个 tag（或文本末尾）的内容
        pat = r"\[" + re.escape(tag_from) + r"\]\s*(.*?)(?:" + "|".join(
            [r"\[" + re.escape(t) + r"\]" for t in tag_to_list]
        ) + r"|$)"
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    pros = _grab("Positive Developments", ["Potential Concerns", "Prediction & Analysis"])
    cons = _grab("Potential Concerns", ["Prediction & Analysis", "Positive Developments"])
    pa   = _grab("Prediction & Analysis", ["Positive Developments", "Potential Concerns"])

    # 从 Prediction & Analysis 里抽 prediction 行，再归一化
    pred_line = ""
    m_pred = re.search(r"prediction\s*:\s*([^\n\r]+)", pa, flags=re.IGNORECASE)
    if m_pred:
        pred_line = m_pred.group(1).strip()
    pred_dir = _normalize_direction(pred_line)

    # 把 Analysis: 后面的正文抽出来（可选）
    analysis = pa
    m_ana = re.search(r"analysis\s*:\s*(.*)", pa, flags=re.IGNORECASE | re.DOTALL)
    if m_ana:
        analysis = m_ana.group(1).strip()

    out = {}
    if pros: out["positive developments"] = pros
    if cons: out["potential concerns"]    = cons
    if analysis: out["analysis"]          = analysis
    out["prediction"] = pred_dir  # 评估方向时可用
    return out

# 之后的计算用我们重写的解析器
def loose_calc_metrics(gts, answers):
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)

    for ans, gt in zip(answers, gts):
        ans_d = parse_answer_relaxed(ans)
        gt_d  = parse_answer_relaxed(gt)

        # 只要 pros/cons/analysis 任意非空就计入
        keys = ["positive developments", "potential concerns", "analysis"]
        has_any = any(k in ans_d and k in gt_d and ans_d[k] and gt_d[k] for k in keys)
        if not has_any:
            continue

        for k in keys:
            if k in ans_d and k in gt_d and ans_d[k] and gt_d[k]:
                answers_dict[k].append(ans_d[k])
                gts_dict[k].append(gt_d[k])

    metrics = {}
    if answers_dict.get("positive developments"):
        metrics["pros_rouge_scores"] = calc_rouge_score(
            gts_dict["positive developments"], answers_dict["positive developments"]
        )
    if answers_dict.get("potential concerns"):
        metrics["cons_rouge_scores"] = calc_rouge_score(
            gts_dict["potential concerns"], answers_dict["potential concerns"]
        )
    if answers_dict.get("analysis"):
        metrics["anal_rouge_scores"] = calc_rouge_score(
            gts_dict["analysis"], answers_dict["analysis"]
        )

    metrics["valid_count"] = len(answers_dict.get("analysis", [])) \
                             or len(answers_dict.get("positive developments", [])) \
                             or len(answers_dict.get("potential concerns", []))
    return metrics

# 重新读你已经有的文件
d_base = pd.read_pickle("./comparison_results/deepseek_base_answers.pkl")
d_ft   = pd.read_pickle("./comparison_results/deepseek_finetuned_answers.pkl")
d_gts  = pd.read_pickle("./comparison_results/deepseek_gts.pkl")

print("base_size:", len(d_base))
print("ft_size:", len(d_ft))
print("gts_size:", len(d_gts))

base_m = loose_calc_metrics(d_gts, d_base)
ft_m   = loose_calc_metrics(d_gts, d_ft)

with open("./comparison_results/deepseek_base_metrics.pkl", "wb") as f:
    pickle.dump(base_m, f)
with open("./comparison_results/deepseek_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(ft_m, f)

print("DeepSeek Base Model Metrics:", base_m)
print("DeepSeek Fine-Tuned Model Metrics:", ft_m)
print("Llama2 Base Model Meterics:", llama2_base)
print("Llama2 ft Metrics:", llama2_ft)

#print(d_base[0],d_ft[0], d_gts[0])