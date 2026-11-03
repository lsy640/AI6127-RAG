
import gradio as gr
import time

from hybrid_retriever import hybrid_search
from generator import (
    format_meta,
    generate_answer,
    load_product_meta,
    product_meta,
    product_title,
)


def product_line(asin, passage):
    """Title and price/rating line for display."""
    title = product_title(passage) or asin
    meta = format_meta(product_meta(asin, passage))
    return title, meta


def search_fn(query):
    """
    Complete RAG pipeline:

    1. Retrieve products using BM25 + FAISS + RRF.
    2. Generate an answer using the retrieved evidence.
    3. Display recommendations and evidence.
    4. Record retrieval and generation performance.
    """

    if not query or not query.strip():
        return (
            "Please enter a product query.",
            "",
            ""
        )

    # ==========================================
    # Step 1: Hybrid Retrieval (Q2)
    # ==========================================

    try:
        results, retrieval_latency = hybrid_search(
            query,
            top_k=5
        )

    except Exception as e:
        return (
            f"Retrieval failed: {str(e)}",
            "",
            ""
        )

    if not results:
        return (
            "No relevant products were retrieved.",
            "",
            f"Retrieval latency: {retrieval_latency} ms"
        )

    # ==========================================
    # Step 2: LLM Generation (Q4)
    # ==========================================

    generation_start = time.perf_counter()

    try:
        generated = generate_answer(
            query,
            results
        )

        generation_latency = generated.get(
            "latency_ms",
            round(
                (time.perf_counter() - generation_start)
                * 1000,
                2
            )
        )

        answer = generated.get(
            "answer",
            "No answer was generated."
        )

        recommendations = generated.get(
            "recommendations",
            []
        )

        # ======================================
        # Step 3: Format generated recommendations
        # ======================================

        answer_text = (
            "==== AI Shopping Assistant Answer ====\n\n"
            f"{answer}\n\n"
        )

        passages = {
            item["parent_asin"]: item["passage"]
            for item in results
        }

        if recommendations:
            answer_text += "==== Recommended Products ====\n\n"

            for index, item in enumerate(
                recommendations,
                start=1
            ):
                title, meta = product_line(
                    item["parent_asin"],
                    passages.get(item["parent_asin"], "")
                )
                answer_text += (
                    f"{index}. {title}\n"
                    f"   ASIN: {item['parent_asin']}"
                    + (f" | {meta}" if meta else "")
                    + "\n"
                )

                answer_text += (
                    f"   Reason: {item['reason']}\n"
                )

                answer_text += (
                    "   Evidence: "
                    + ", ".join(item["evidence_ids"])
                    + "\n\n"
                )

        else:
            answer_text += (
                "No products could be safely recommended "
                "from the retrieved evidence.\n"
            )

        if generated.get("insufficient_evidence"):
            answer_text += (
                "\nNote: The retrieved evidence may be "
                "insufficient to fully answer this question.\n"
            )

        # ======================================
        # Step 4: Display retrieved evidence
        # ======================================

        evidence_text = (
            "==== Top 5 Retrieved Product Passages ====\n\n"
        )

        for index, item in enumerate(
            results,
            start=1
        ):
            title, meta = product_line(
                item["parent_asin"], item["passage"]
            )
            evidence_text += (
                f"E{index} | {title}\n"
                f"ASIN: {item['parent_asin']}"
                + (f" | {meta}" if meta else "")
                + "\n"
                + (
                    f"Matched: {item['matched_field']} "
                    f"(BM25 rank {item.get('bm25_rank') or '-'}, "
                    f"dense rank {item.get('dense_rank') or '-'})\n"
                    if item.get("matched_field") else ""
                )
            )

            evidence_text += (
                f"{item['passage']}\n\n"
            )

        # ======================================
        # Step 5: Performance information
        # ======================================

        prompt_tokens = generated.get(
            "prompt_tokens",
            0
        )

        completion_tokens = generated.get(
            "completion_tokens",
            0
        )

        model = generated.get(
            "model",
            "Unknown"
        )

        metrics_text = (
            "==== System Performance ====\n\n"
            f"Retrieval latency: {retrieval_latency} ms\n"
            f"Generation latency: {generation_latency} ms\n"
            f"Total latency: "
            f"{round(retrieval_latency + generation_latency, 2)} ms\n\n"
            f"LLM model: {model}\n"
            f"Prompt tokens: {prompt_tokens}\n"
            f"Completion tokens: {completion_tokens}\n"
            f"Total tokens: {prompt_tokens + completion_tokens}\n"
        )

        return (
            answer_text,
            evidence_text,
            metrics_text
        )

    except Exception as e:

        # Retrieval still works even if LLM generation fails.
        evidence_text = (
            "==== Retrieved Product Evidence ====\n\n"
        )

        for index, item in enumerate(
            results,
            start=1
        ):
            evidence_text += (
                f"Rank {index} | "
                f"ASIN: {item['parent_asin']}\n"
                f"{item['passage']}\n\n"
            )

        error_text = (
            "LLM generation failed.\n\n"
            f"Error: {str(e)}\n\n"
            "Please check your API configuration."
        )

        metrics_text = (
            f"Retrieval latency: {retrieval_latency} ms\n"
            "Generation latency: Failed\n"
        )

        return (
            error_text,
            evidence_text,
            metrics_text
        )


# ==============================================
# Gradio Web Interface
# ==============================================

with gr.Blocks(
    title="Amazon Product Hybrid RAG System"
) as demo:

    gr.Markdown(
        """
        # Amazon Product Hybrid RAG System

        ### BM25 + FAISS + RRF + LLM Generation

        This system retrieves Amazon product information
        from the local knowledge base and generates
        evidence-grounded shopping recommendations.
        """
    )

    query = gr.Textbox(
        label="Input your shopping query",
        placeholder=(
            "e.g. waterproof men's hiking boots "
            "with good comfort"
        ),
        lines=2
    )

    search_button = gr.Button(
        "Search and Generate Answer",
        variant="primary"
    )

    gr.Markdown("## AI Shopping Assistant")

    answer_output = gr.Textbox(
        label="Generated Answer and Recommendations",
        lines=12
    )

    with gr.Accordion(
        "Retrieved Product Evidence",
        open=True
    ):
        evidence_output = gr.Textbox(
            label="Top 5 Retrieved Passages",
            lines=18
        )

    with gr.Accordion(
        "System Performance",
        open=False
    ):
        metrics_output = gr.Textbox(
            label="Retrieval and Generation Metrics",
            lines=8
        )

    search_button.click(
        fn=search_fn,
        inputs=[query],
        outputs=[
            answer_output,
            evidence_output,
            metrics_output
        ]
    )

    query.submit(
        fn=search_fn,
        inputs=[query],
        outputs=[
            answer_output,
            evidence_output,
            metrics_output
        ]
    )


if __name__ == "__main__":
    # Warm up the embedding model and metadata so the first user query
    # is not charged with one-off loading time.
    print("Warming up retriever and product metadata ...")
    hybrid_search("warm up", top_k=1)
    load_product_meta()
    demo.launch()
