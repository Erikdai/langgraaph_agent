import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os
import requests

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
            return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}}
    except:
        return {"parsed_info": {"国家": "未知", "行业": "未知", "备注": content}}

def report_generator(state: State) -> dict:
    parsed = state["parsed_info"]
    prompt = [
        {"role": "system", "content": (
            f"请基于以下信息撰写一份出海建议报告，结构包含：\n"
            f"1. 企业背景：{parsed}\n"
            "2. 市场分析：请根据该企业背景推测其在目标市场可能面临的机会与挑战。\n"
            "要求内容连贯，结构清晰，语言专业，不少于150字，使用中文撰写。"
        )}
    ]
    return {"report": call_groq_llm(prompt)}

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

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)

    user_prompt = st.chat_input("请输入您的企业出海背景或提问…")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append(("user", user_prompt))

        with st.spinner("正在生成出海建议…"):
            result = app.invoke({"user_input": user_prompt})
            report = result["report"]

        st.chat_message("assistant").markdown(report)
        st.session_state.chat_history.append(("assistant", report))

if __name__ == "__main__":
    main()
