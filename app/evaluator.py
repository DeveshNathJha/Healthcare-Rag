"""
evaluator.py — LLM-as-Judge RAG Evaluation Module
====================================================
WHY THIS FILE EXISTS:
  Keyword-based evaluation breaks on medical synonyms — "fracture" vs "broken bone",
  "pyrexia" vs "fever", "BP" vs "blood pressure". A senior AI engineer solution is to
  use the LLM itself as an impartial judge: pass it the Question + Context + Answer
  and ask it to score each dimension on 0.0–1.0.

  This is the same principle as RAGAS, but using our *existing* Groq 8B model
  (already initialised in rag_chain.py) instead of OpenAI — so it's:
    - Semantic (meaning-based, not keyword-based)
    - Free of extra dependencies
    - Practically free in cost (~$0.0001 per eval call with 8B model)
    - Gracefully degradable (API error → silent null fallback, query never fails)

METRICS COMPUTED (in a SINGLE Groq call):
  1. faithfulness       — Does the answer ONLY use info from the retrieved context?
                          High faithfulness = low hallucination.
  2. answer_relevance   — Does the answer directly address the user's question?
  3. context_precision  — Was the retrieved context actually useful for this question?
                          Low precision = FAISS retrieved irrelevant chunks.

EVAL GRADE (composite):
  A  ≥ 0.85 average   → Excellent
  B  ≥ 0.70 average   → Good
  C  ≥ 0.50 average   → Fair
  F  <  0.50 average  → Poor

DESIGN CHOICES:
  - Judge uses llama-3-8b-8192 (NOT 70B) — yes/no scoring doesn't need heavy reasoning.
  - Single structured prompt → JSON output: 1 API call for all 3 metrics.
  - `temperature=0` for deterministic, reproducible scores.
  - If judge call fails for any reason, returns None scores (never blocks main query).
"""

import json
import time
import re
from typing import Optional, Dict, Any

from langchain_groq import ChatGroq
from app.utils import get_logger

logger = get_logger(__name__)

# ── JUDGE MODEL ───────────────────────────────────────────────────────────────
# WHY 8B, not 70B:
#   Evaluation is a structured yes/no scoring task — it does not need heavy
#   reasoning. 8B is 10x cheaper and ~3x faster while producing equivalent
#   scores for binary grounding checks.
JUDGE_MODEL = "llama-3-8b-8192"

# ── GRADE THRESHOLDS ──────────────────────────────────────────────────────────
GRADE_THRESHOLDS = {
    "A": 0.85,
    "B": 0.70,
    "C": 0.50,
}

# ── JUDGE PROMPT ──────────────────────────────────────────────────────────────
# WHY this exact wording:
#   "fracture=broken bone=OK" in the prompt explicitly teaches the judge that
#   medical synonym equivalence should NOT penalise faithfulness. Without this,
#   even a smart 8B model might over-penalise paraphrasing.
JUDGE_PROMPT_TEMPLATE = """You are an impartial RAG (Retrieval-Augmented Generation) evaluation judge for a healthcare AI system.

Your task is to evaluate the quality of an AI-generated answer against the retrieved context and the original question.

=== QUESTION ===
{question}

=== RETRIEVED CONTEXT ===
{context}

=== GENERATED ANSWER ===
{answer}

=== EVALUATION CRITERIA ===
Score each dimension from 0.0 to 1.0:

1. faithfulness (0.0–1.0):
   - Does the answer ONLY use information present in the retrieved context?
   - Medical synonyms are acceptable (e.g., "fracture" = "broken bone", "pyrexia" = "fever", "BP" = "blood pressure").
   - Score 1.0 = every claim is grounded in context. Score 0.0 = answer is completely hallucinated.

2. answer_relevance (0.0–1.0):
   - Does the answer directly and completely address the user's question?
   - Score 1.0 = perfectly on-topic and complete. Score 0.0 = off-topic or empty.

3. context_precision (0.0–1.0):
   - Was the retrieved context actually useful and relevant for answering this question?
   - Score 1.0 = context was highly relevant. Score 0.0 = context was completely irrelevant to the question.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object with no additional text, no markdown, no explanation:
{{"faithfulness": <float>, "answer_relevance": <float>, "context_precision": <float>}}"""


class RAGEvaluator:
    """
    LLM-as-Judge evaluator for the Healthcare RAG pipeline.

    Usage (from rag_chain.py):
        evaluator = RAGEvaluator(llm_api_key_env="GROQ_API_KEY")
        metrics = evaluator.evaluate(
            question="What are symptoms of diabetes?",
            context="Diabetes causes increased thirst, frequent urination...",
            answer="Symptoms include polyuria and polydipsia..."
        )
        # Returns: {"faithfulness": 0.9, "answer_relevance": 0.95,
        #           "context_precision": 0.85, "eval_grade": "A",
        #           "judge_model": "llama-3-8b-8192", "eval_latency_ms": 340}
    """

    def __init__(self):
        """
        Initialise the judge LLM.
        Uses the same GROQ_API_KEY already set in the environment by rag_chain.py.
        WHY temperature=0: Evaluation scoring must be deterministic and reproducible.
        """
        logger.info(f"[EVALUATOR] Initialising LLM-as-Judge ({JUDGE_MODEL})...")
        self.judge_llm = ChatGroq(
            model_name=JUDGE_MODEL,
            temperature=0,
            max_tokens=150,   # JSON response is tiny — cap tokens to save cost
        )
        logger.info("[EVALUATOR] RAGEvaluator ready.")

    def _compute_grade(self, scores: Dict[str, float]) -> str:
        """
        Computes a composite letter grade from the three metric scores.
        Uses simple average — all three dimensions are equally important.
        """
        avg = sum(scores.values()) / len(scores)
        for grade, threshold in GRADE_THRESHOLDS.items():
            if avg >= threshold:
                return grade
        return "F"

    def _parse_judge_response(self, raw: str) -> Optional[Dict[str, float]]:
        """
        Robustly parse the judge's JSON response.

        WHY robust parsing:
          LLMs sometimes wrap JSON in markdown (```json ... ```) even when
          instructed not to. This extracter handles both cases.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting just the JSON object via regex as last resort
            match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("[EVALUATOR] Could not parse judge JSON response.")
                    return None
            else:
                logger.warning("[EVALUATOR] No JSON found in judge response.")
                return None

        # Validate all required keys exist and are floats in [0, 1]
        required_keys = {"faithfulness", "answer_relevance", "context_precision"}
        if not required_keys.issubset(data.keys()):
            logger.warning(f"[EVALUATOR] Missing keys in judge response: {data}")
            return None

        # Clamp values to [0.0, 1.0] in case model returns out-of-range
        return {
            k: round(max(0.0, min(1.0, float(data[k]))), 3)
            for k in required_keys
        }

    def evaluate(
        self,
        question: str,
        context: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Main evaluation method — calls the judge LLM and returns scored metrics.

        PARAMETERS:
          question : The original user query.
          context  : The concatenated retrieved context string (from expanded_docs).
          answer   : The LLM-generated answer string.

        RETURNS:
          On success:
            {
              "faithfulness": 0.91,
              "answer_relevance": 0.88,
              "context_precision": 0.80,
              "eval_grade": "A",
              "judge_model": "llama-3-8b-8192",
              "eval_latency_ms": 340
            }
          On failure (API error, timeout, bad JSON):
            {
              "faithfulness": None,
              "answer_relevance": None,
              "context_precision": None,
              "eval_grade": "N/A",
              "judge_model": "llama-3-8b-8192",
              "eval_latency_ms": 0
            }

        WHY returns None on failure instead of raising:
          The main /query endpoint must NEVER fail due to evaluation.
          If Groq has a hiccup, the user still gets their answer — eval is a bonus.
        """
        t_eval_start = time.perf_counter()

        # Truncate context to avoid exceeding judge's context window
        # WHY 3000 chars: Judge only needs a summary-level view of context, not all 6000 tokens
        context_for_judge = context[:3000] if len(context) > 3000 else context

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context=context_for_judge,
            answer=answer
        )

        NULL_RESULT = {
            "faithfulness":      None,
            "answer_relevance":  None,
            "context_precision": None,
            "eval_grade":        "N/A",
            "judge_model":       JUDGE_MODEL,
            "eval_latency_ms":   0
        }

        try:
            raw_response = self.judge_llm.invoke(prompt)
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

            scores = self._parse_judge_response(raw_text)

            if scores is None:
                logger.warning("[EVALUATOR] Returning null metrics due to parse failure.")
                return NULL_RESULT

            eval_latency_ms = round((time.perf_counter() - t_eval_start) * 1000, 1)
            grade = self._compute_grade(scores)

            logger.info(
                f"[EVALUATOR] Scores → faithfulness={scores['faithfulness']} | "
                f"relevance={scores['answer_relevance']} | "
                f"precision={scores['context_precision']} | "
                f"grade={grade} | latency={eval_latency_ms}ms"
            )

            return {
                "faithfulness":      scores["faithfulness"],
                "answer_relevance":  scores["answer_relevance"],
                "context_precision": scores["context_precision"],
                "eval_grade":        grade,
                "judge_model":       JUDGE_MODEL,
                "eval_latency_ms":   eval_latency_ms
            }

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t_eval_start) * 1000, 1)
            logger.warning(
                f"[EVALUATOR] Judge call failed after {elapsed_ms}ms — "
                f"{type(exc).__name__}: {exc}. Returning null metrics."
            )
            return NULL_RESULT
