"""
Gradio 多轮对话界面：澄清式 RAG 商品推荐助手

左侧聊天：用户输入需求（中文或英文），助手会先反问补全需求，再给出有据推荐。
右侧面板：当前追踪到的需求（对话状态）、检索证据（命中字段、价格、评分）、本轮耗时与 LLM 调用。
可切换 Agent 配置（full / 基线 B0 等），用于演示消融实验。

运行：python app.py   （需设置 DEEPSEEK_API_KEY、DEEPSEEK_MODEL）
"""

import json

import gradio as gr

import llm
from agent import CONFIGS, Agent
from generator import format_meta, product_meta

EXAMPLES = [
    "I need some shoes.",
    "推荐一个适合 3 岁孩子的玩具",
    "A non-stick frying pan, under $50.",
    "I'm looking for a jigsaw puzzle for a 10-year-old under $25.",
]


def respond(message, chat, history, config_name):
    if not message or not message.strip():
        return chat, history, "", "", "", ""
    history = history + [{"role": "user", "text": message.strip()}]
    usage = llm.Usage()
    agent = Agent(CONFIGS[config_name], usage=usage)
    try:
        out = agent.respond(history)
    except Exception as error:
        chat = chat + [{"role": "user", "content": message},
                       {"role": "assistant", "content": f"Error: {error}"}]
        return chat, history, "", "", "", ""

    history = history + [{"role": "assistant", "text": out.text, "action": out.action,
                          "asked_slot": out.asked_slot}]
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": out.text}]

    state = {k: v for k, v in out.state.items() if k not in ("reason",)}
    state_md = "```json\n" + json.dumps(state, ensure_ascii=False, indent=2) + "\n```"

    if out.evidence:
        ev_lines = []
        for e in out.evidence:
            ev_lines.append(f"**{e['evidence_id']}** {e['title']}  \n"
                            f"ASIN {e['parent_asin']} · {format_meta(product_meta(e['parent_asin']))}  \n"
                            f"{e['passage'][:400]}…")
        evidence_md = "\n\n".join(ev_lines)
    else:
        evidence_md = (f"Action: **{out.action}**" +
                       (f" — asked about **{out.asked_slot}**" if out.asked_slot else ""))
        ent = out.debug.get("entropies") or {}
        if ent:
            evidence_md += "\n\nCandidate entropy per missing attribute:\n" + "\n".join(
                f"- {k}: {v[0]} (top values: {', '.join(map(str, v[1]))})" for k, v in ent.items())

    d = out.debug
    perf_md = (f"- Turn latency: {d.get('latency_ms', 0):.0f} ms\n"
               f"- Retrieval: {d.get('retrieval_ms') or 0:.0f} ms (relaxed: {d.get('relaxed') or 'none'})\n"
               f"- Search query: `{d.get('query')}`\n"
               f"- LLM calls: {usage.calls}, tokens: {usage.prompt_tokens + usage.completion_tokens}\n"
               + (f"- Evidence check: {d['evidence_check']}\n" if d.get("evidence_check") else "")
               + (f"- Self-check: {d['self_check']}\n" if d.get("self_check") else ""))
    return chat, history, "", state_md, evidence_md, perf_md


def reset():
    return [], [], "", "", "", ""


with gr.Blocks(title="Conversational RAG Shopping Assistant") as demo:
    gr.Markdown("# Conversational RAG Shopping Assistant\n"
                "Hybrid retrieval (BM25 + bge-m3 + RRF + re-ranking) over 15,000 Amazon products · "
                "asks clarifying questions when your need is incomplete · grounded, cited recommendations.")
    history = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(type="messages", height=520, label="Assistant")
            box = gr.Textbox(placeholder="Describe what you are looking for (English or 中文)…",
                             label="Your message", lines=2)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("New conversation")
            config = gr.Dropdown(list(CONFIGS), value="full", label="Agent configuration (ablation)")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            with gr.Accordion("Tracked need (dialogue state)", open=True):
                state_out = gr.Markdown()
            with gr.Accordion("Retrieved evidence", open=True):
                evidence_out = gr.Markdown()
            with gr.Accordion("Performance", open=False):
                perf_out = gr.Markdown()

    outputs = [chat, history, box, state_out, evidence_out, perf_out]
    send.click(respond, [box, chat, history, config], outputs)
    box.submit(respond, [box, chat, history, config], outputs)
    clear.click(reset, None, outputs)


if __name__ == "__main__":
    # 预热检索与重排序模型，避免第一位用户等待模型加载
    print("Warming up retriever and re-ranker ...")
    import hybrid_retriever as hr
    hr.search("warm up", top_k=3, rerank=True)
    demo.launch()
