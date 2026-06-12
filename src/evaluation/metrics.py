import logging

logger = logging.getLogger(__name__)


def _bertscore(predictions: list[str], references: list[str],
               model_type: str, device: str | None) -> dict:
    try:
        from bert_score import score as _score
    except ImportError:
        logger.warning(
            "[Evaluation] bert-score not installed — BERTScore skipped. "
            "Install with: pip install bert-score"
        )
        return {}
    P, R, F1 = _score(predictions, references,
                       model_type=model_type, device=device, verbose=False)
    return {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall":    round(float(R.mean()), 4),
        "bertscore_f1":        round(float(F1.mean()), 4),
        "bertscore_model":     model_type,
    }


def _rouge(predictions: list[str], references: list[str]) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning(
            "[Evaluation] rouge-score not installed — ROUGE skipped. "
            "Install with: pip install rouge-score"
        )
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(result[k].fmeasure)
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def evaluate(predictions: list[str], references: list[str],
             bertscore_model: str = "distilbert-base-uncased",
             device: str | None = None) -> dict:
    """
    Compute BERTScore F1 and ROUGE on prediction/reference pairs.

    Both scoring functions are imported lazily — if a package is not installed
    that scorer is skipped and a warning is logged; the other still runs.

    Returns a flat dict of scalars ready to merge into the training metrics JSON.
    Returns {} if predictions is empty.
    """
    if not predictions:
        return {}
    scores: dict = {"eval_samples": len(predictions)}
    scores.update(_bertscore(predictions, references, bertscore_model, device))
    scores.update(_rouge(predictions, references))
    return scores
