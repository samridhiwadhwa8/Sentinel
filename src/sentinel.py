from src.retrieve import retrieve
from src.llm import answer_question
from src.faithfulness import FAITHFULNESS_THRESHOLD, check_faithfulness


def sentinel(question: str, top_k: int = 3):
    # Clean question text
    effective_question = (question or "").strip()

    # Retrieve from MedRAG textbooks
    all_chunks = retrieve(effective_question, top_k=top_k)

    if not all_chunks:
        return {
            "question": effective_question,
            "retrieved_chunks": [],
            "decision": "ABSTAIN",
            "reason": "No medical evidence was retrieved.",
            "faithfulness_probability": 0.0,
            "unfaithfulness_probability": 1.0,
            "classifier_label": "UNFAITHFUL",
        }

    answer = answer_question(
        effective_question,
        all_chunks
    )

    context_parts = []
    for number, chunk in enumerate(all_chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        title = str(chunk.get("title", "unknown source"))
        if text:
            context_parts.append(
                f"[Source {number}: {title}]\n{text}"
            )

    context = "\n\n".join(context_parts)

    faithfulness = check_faithfulness(
        context=context,
        statement=answer
    )

    is_abstain_text = (
        "not have enough evidence" in answer.lower()
        or "does not appear in the verified medical" in answer.lower()
    )

    all_low_confidence = all(
        c.get("confidence") == "low" or c.get("is_out_of_domain")
        for c in all_chunks
    )

    if (
        not is_abstain_text
        and not all_low_confidence
        and faithfulness["faithfulness_probability"] >= FAITHFULNESS_THRESHOLD
    ):
        decision = "ANSWER"
        reason = None
    else:
        decision = "ABSTAIN"
        faithfulness["faithfulness_probability"] = min(faithfulness["faithfulness_probability"], 0.15)
        faithfulness["unfaithfulness_probability"] = round(1.0 - faithfulness["faithfulness_probability"], 3)
        faithfulness["label"] = "UNFAITHFUL"
        if is_abstain_text or all_low_confidence:
            reason = (
                "Non-medical or out-of-domain query. Verified MedRAG textbooks contain no evidence. "
                "Sentinel automatically abstains to prevent ungrounded generation."
            )
        else:
            pct = round(faithfulness["faithfulness_probability"] * 100)
            thresh = int(FAITHFULNESS_THRESHOLD * 100)
            reason = (
                f"Evidence alignment ({pct}%) falls below the {thresh}% safety threshold. "
                "Sentinel automatically abstains to protect clinical safety."
            )

    return {
        "question": effective_question,
        "retrieved_chunks": all_chunks,
        "answer": answer,
        "faithfulness_probability":
            faithfulness["faithfulness_probability"],
        "unfaithfulness_probability":
            faithfulness["unfaithfulness_probability"],
        "classifier_label":
            faithfulness["label"],
        "decision": decision,
        "reason": reason,
    }
