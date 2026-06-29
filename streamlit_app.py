"""
RAG 文档问答 —— Streamlit 可视化界面

启动：streamlit run streamlit_app.py
"""
import streamlit as st
from rag_core import build_collection, get_collection, ask

st.set_page_config(page_title="RAG 文档问答", page_icon="📚", layout="wide")
st.title("📚 RAG 文档问答机器人")

# ====== 侧边栏：文档上传 ======
with st.sidebar:
    st.header("📄 上传文档")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["pdf", "txt"])

    if uploaded_file is not None:
        with st.spinner("正在处理文档..."):
            if uploaded_file.name.endswith(".pdf"):
                # PDF 需要临时保存再读取
                import tempfile
                from rag_core import load_pdf_text
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                text = load_pdf_text(tmp_path)
                import os
                os.unlink(tmp_path)
            else:
                text = uploaded_file.read().decode("utf-8")

            col, count = build_collection(text, "rag_ui")
            st.session_state["collection"] = col
            st.success(f"处理完成！切分为 {count} 个片段")

    # 或直接粘贴文本
    st.divider()
    st.caption("或者直接粘贴文本：")
    pasted_text = st.text_area("文档内容", height=150, placeholder="在此粘贴文档内容...")
    if st.button("建库", use_container_width=True) and pasted_text.strip():
        with st.spinner("正在处理..."):
            col, count = build_collection(pasted_text.strip(), "rag_ui")
            st.session_state["collection"] = col
            st.success(f"完成！{count} 个片段")

    # 加载已有库
    st.divider()
    if st.button("📂 加载已有向量库", use_container_width=True):
        try:
            st.session_state["collection"] = get_collection("pdf_knowledge")
            st.success(f"已加载，{st.session_state['collection'].count()} 条记录")
        except:
            st.warning("未找到已有向量库")

# ====== 主区域：问答界面 ======
col = st.session_state.get("collection")

if col is None:
    st.info("👈 先在左侧上传文档或粘贴文本，也可以点击'加载已有向量库'")
else:
    # 显示向量库状态
    st.caption(f"向量库就绪 · {col.count()} 条记录")

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 显示历史消息
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for s in msg["sources"]:
                        st.caption(f"距离={s['distance']} | {s['text_preview']}")

    # 输入框
    if question := st.chat_input("输入你的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["messages"].append({"role": "user", "content": question})

        # 生成答案
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = ask(col, question)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📎 参考来源"):
                    for s in result["sources"]:
                        st.caption(f"距离={s['distance']} | {s['text_preview']}")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
