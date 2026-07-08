"""
Answer-quality metrics:
 - EM / F1 against a gold answer (simple, deterministic)
 - Faithfulness + relevance via LLM-as-judge (Ollama), since "is this
   grounded in the context" isn't something EM/F1 can capture.
"""
import re
import json
import requests
from app import config


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


JUDGE_PROMPT = """You are a strict evaluator. Given a QUESTION, the CONTEXT that was
retrieved, and an ANSWER, score the answer on two dimensions from 1 (worst) to 5 (best):

- faithfulness: is every claim in the answer actually supported by the CONTEXT?
  (5 = fully grounded, no unsupported claims; 1 = mostly fabricated / ignores context)
- relevance: does the answer actually address the QUESTION asked?
  (5 = fully answers it; 1 = off-topic or non-answer)

Respond with ONLY a JSON object, no other text, in this exact format:
{{"faithfulness": <1-5>, "relevance": <1-5>, "rationale": "<one sentence>"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}
"""


def llm_judge(question: str, context: str, answer: str) -> dict:
    """
    Calls the local Ollama model as a judge. Returns a dict with
    faithfulness (1-5), relevance (1-5), rationale, and a flag if
    JSON parsing failed (robust handling of malformed judge output).
    """
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=600,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # models sometimes wrap JSON in markdown fences - strip those
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        print("Parsed:", parsed)
        print("Type of relevance:", type(parsed.get("relevance")))
        print("Value of relevance:", parsed.get("relevance"))
        return {
            "faithfulness": float(parsed.get("faithfulness", 0)),
            "relevance": float(parsed.get("relevance", 0)),
            "rationale": parsed.get("rationale", ""),
            "parse_ok": True,
        }
    except (json.JSONDecodeError, requests.RequestException, KeyError, ValueError) as e:
        return {
            "faithfulness": None,
            "relevance": None,
            "rationale": f"judge call/parse failed: {e}",
            "parse_ok": False,
        }
