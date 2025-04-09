from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os
import requests
import matplotlib.pyplot as plt
import uuid

os.environ["GROQ_API_KEY"] = "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"

class State(TypedDict):
    user_input: str
    parsed_info: Optional[dict]
    search_result: Optional[str]
    report: Optional[str]
    history: Optional[str]
    chart_path: Optional[str]

def call_groq_llm(prompt_messages):
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": prompt_messages,
        "temperature": 0.7
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"[Groq API 错误] {response.status_code}: {response.text}"

def recognition_user_input(state: State) -> dict:
    user_input = state['user_input']
    prompt = [
        {"role": "system", "content": (
            "你是一个出海顾问助手，需要用户提供完整的企业出海背景后才能进行分析。\n"
            "请提取以下字段：\n"
            "- 出海国家或地区（必填）\n"
            "- 企业主营业务或行业（必填）\n"
            "请以 JSON 格式返回字段值。"
        )},
        {"role": "user", "content": user_input}
    ]
    content = call_groq_llm(prompt)
    try:
        parsed = eval(content)
        if isinstance(parsed, dict) and '国家' in parsed and '行业' in parsed:
            return {"parsed_info": parsed}
        else:
            return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}}
    except:
        return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}}

def web_search_agent(state: State) -> dict:
    parsed = state["parsed_info"]
    keyword = f"{parsed.get('国家', '')} {parsed.get('行业', '')} 出海政策 市场趋势"
    headers = {"Authorization": "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", "Content-Type": "application/json"}
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json={"query": keyword, "include_raw_content": True}
        )
        print("[Tavily 调试] 请求关键词：", keyword)
        print("[Tavily 调试] 返回状态码：", response.status_code)
        print("[Tavily 调试] 返回内容：", response.text)

        if response.ok and response.json().get("results"):
            content_summary = response.json()['results'][0]['content']
        else:
            content_summary = "未能完成联网搜索，请稍后再试。"
    except Exception as e:
        print("[Tavily 异常]", e)
        content_summary = f"[模拟内容] {parsed.get('国家')} 市场的 {parsed.get('行业')} 行业目前出海活跃，有政府支持、电商平台扩展等趋势。"

    analysis_prompt = [
        {"role": "system", "content": (
            "你是一个出海市场分析助手，请根据以下网页内容，总结适用于该国家与行业的市场趋势、合规重点或风险警示，控制在200字以内。\n\n"
            f"网页内容：{content_summary}"
        )}
    ]
    summary = call_groq_llm(analysis_prompt)
    return {"search_result": summary}


def report_generator(state: State) -> dict:
    parsed = state["parsed_info"]
    insight = state["search_result"]
    chart = state.get("chart_path", "(图表未生成)")
    prompt = [
        {"role": "system", "content": (
            f"请基于以下信息撰写一份出海建议报告，结构包含：\n"
            f"1. 企业背景：{parsed}\n"
            f"2. 市场分析：{insight}\n"
            "要求内容连贯，结构清晰，语言专业，不少于150字，使用中文撰写。"
        )}
    ]
    return {"report": call_groq_llm(prompt)}

graph = StateGraph(State)
graph.add_node("parse", recognition_user_input)
graph.add_node("search", web_search_agent)
graph.add_node("report_gen", report_generator)

graph.set_entry_point("parse")
graph.add_edge("parse", "search")
graph.add_edge("search", "report_gen")
graph.add_edge("report_gen", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "user_input": "我们是一家从事建材出口的公司，考虑2025年拓展中东市场，请给我出海建议。"
    })
    print("=== 出海建议报告 ===")
    print(result["report"])
    print(result.keys())
