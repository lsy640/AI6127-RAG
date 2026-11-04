"""
DeepSeek（OpenAI 兼容接口）调用封装：JSON 输出、重试、用量统计。

思考模式：deepseek-flash 默认开启思考（reasoning），此时 temperature 被忽略、
输出不可复现，且每次多出数百个推理 token。本项目默认关闭思考
（extra_body={"thinking": {"type": "disabled"}}），需要时可按调用开启，
用于 Q5 中“推理模式 vs 非推理模式”的对比。

环境变量：DEEPSEEK_API_KEY、DEEPSEEK_MODEL、DEEPSEEK_THINKING（disabled|enabled，默认 disabled）
"""

import os
import threading
import time

from generator import create_client, get_model_name, parse_json_response

DEFAULT_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled") == "enabled"


class Usage:
    """线程安全的调用计数（按 tag 分组），用于成本与延迟统计。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.by_tag = {}

    def add(self, tag, prompt, completion, seconds):
        with self.lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.seconds += seconds
            t = self.by_tag.setdefault(tag, [0, 0, 0])
            t[0] += 1
            t[1] += prompt
            t[2] += completion

    def snapshot(self):
        with self.lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "llm_seconds": round(self.seconds, 2),
                "by_tag": {k: {"calls": v[0], "prompt_tokens": v[1], "completion_tokens": v[2]}
                           for k, v in self.by_tag.items()},
            }


USAGE = Usage()


def thinking_kwargs(thinking):
    return {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}


def chat(messages, json_mode=True, max_tokens=800, temperature=0, thinking=None,
         model=None, tag="other", retries=2, usage=None):
    """调用一次模型；json_mode 时返回 dict，否则返回字符串。"""
    thinking = DEFAULT_THINKING if thinking is None else thinking
    usage = usage or USAGE
    client = create_client()
    kwargs = dict(model=model or get_model_name(), messages=messages,
                  max_tokens=max_tokens + (2000 if thinking else 0),
                  temperature=temperature, **thinking_kwargs(thinking))
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            dt = time.perf_counter() - t0
            u = resp.usage
            usage.add(tag, u.prompt_tokens if u else 0, u.completion_tokens if u else 0, dt)
            content = resp.choices[0].message.content or ""
            return parse_json_response(content) if json_mode else content.strip()
        except Exception as error:   # 网络错误、JSON 解析失败等：重试
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")
