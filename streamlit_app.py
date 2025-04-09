import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os
import requests
import re

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

class State(TypedDict):
    user_input: str
    parsed_info: Optional[dict]
    report: Optional[str]

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
            return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}, "debug": content}
    except:
        return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}, "debug": content}

def report_generator(state: State) -> dict:
    parsed = state["parsed_info"]
    prompt = [
        {"role": "system", "content": (
            f"请基于以下信息撰写一份出海建议报告，结构包含：\n"
            f"1. 企业背景：{parsed}\n"
            "2. 市场分析：请根据该企业背景推测其在目标市场可能面临的机会与挑战。\n"
            "要求内容连贯，结构清晰，语言专业，不少于150字，使用中文撰写。并在回答中使用 <think>你的思考过程</think> 标签包装推理内容。"
        )}
    ]
    full_output = call_groq_llm(prompt)
    # 抽取推理过程
    match = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
    debug_info = match.group(1).strip() if match else "（无推理过程标注）"
    cleaned_output = re.sub(r"<think>.*?</think>", "", full_output, flags=re.DOTALL).strip()
    return {"report": cleaned_output, "debug": debug_info}

graph = StateGraph(State)
graph.add_node("parse", recognition_user_input)
graph.add_node("report_gen", report_generator)

graph.set_entry_point("parse")
graph.add_edge("parse", "report_gen")
graph.add_edge("report_gen", END)

app = graph.compile()

def main():
    st.set_page_config(page_title="出海顾问助手", page_icon="🌍")
    st.title("🌍 企业出海智能对话助手")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "full_trace" not in st.session_state:
        st.session_state.full_trace = []

    with st.sidebar:
        if st.button("🧹 清空对话"):
            st.session_state.clear()
            st.rerun()

    for i, (role, msg, trace) in enumerate(st.session_state.full_trace):
        with st.chat_message(role):
            st.markdown(msg)
            if role == "assistant" and trace:
                with st.expander("推理过程展开", expanded=False):
                    st.code(trace)

    user_prompt = st.chat_input("请输入您的企业出海背景或提问…")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.full_trace.append(("user", user_prompt, None))

        with st.spinner("正在生成出海建议…"):
            result = app.invoke({"user_input": user_prompt})
            report = result["report"]
            debug = result.get("debug", "无推理内容")

        st.chat_message("assistant").markdown(report)
        st.session_state.full_trace.append(("assistant", report, debug))

if __name__ == "__main__":
    main()
