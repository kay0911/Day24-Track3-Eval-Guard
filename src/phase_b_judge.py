from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác (accuracy), độ đầy đủ (completeness), và tính súc tích (conciseness).
Trả lời bằng JSON (chỉ trả về chuỗi JSON hợp lệ, không chứa thêm bất kỳ văn bản nào khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn lý do chọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    from src.gemini_client import generate_gemini
    
    prompt = PROMPT_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b)
    resp_text = generate_gemini(
        prompt, 
        system_instruction="Bạn là expert đánh giá RAG. Chỉ trả lời JSON.", 
        json_mode=True
    )
    
    try:
        data = json.loads(resp_text)
        if "winner" not in data or data["winner"] not in {"A", "B", "tie"}:
            data["winner"] = "tie"
        if "reasoning" not in data:
            data["reasoning"] = ""
        if "scores" not in data:
            data["scores"] = {"A": 0.5, "B": 0.5}
        return data
    except Exception as e:
        print(f"Error parsing pairwise judge response: {e}. Raw: {resp_text}")
        return {"winner": "tie", "reasoning": "Failed to parse JSON", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(human_labels, judge_labels))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
                "position_bias_count": 0, "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0}, "interpretation": "No data"}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
                              "b_wins_b_longer": b_wins_b_longer,
                              "total_decisive": decisive},
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import TEST_SET_PATH
    import dataclasses

    # --- Load datasets ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Human labels loaded: {len(human_labels)} questions")

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    gt_map = {item["question"]: item["ground_truth"] for item in test_set}

    # In production: run judge on the same 10 questions to get judge_labels
    judge_results = []
    judge_labels = []

    print("\nRunning LLM-as-Judge on 10 comparison questions...")
    for idx, item in enumerate(human_data):
        q = item["question"]
        model_ans = item["model_answer"]
        gt_ans = gt_map.get(q, "Không có ground truth.")

        # Let the judge compare model_answer (A) against ground_truth (B)
        res = swap_and_average(q, model_ans, gt_ans)
        judge_results.append(res)

        # Map to label: 1 if model_answer is better/tie, 0 if ground_truth is better
        label = 1 if res.final_winner in {"A", "tie"} else 0
        judge_labels.append(label)

        print(f"  [{idx+1}/10] Q: {q[:40]}... Winner: {res.final_winner} (Position consistent: {res.position_consistent})")

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ: {kappa:.3f}")

    bias = bias_report(judge_results)
    print(f"Bias report: {bias}")

    # --- Save Phase B results ---
    report = {
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
        "results": [dataclasses.asdict(r) for r in judge_results]
    }
    
    output_path = "reports/judge_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {output_path}")
