import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os
import matplotlib.pyplot as plt

# Supported file formats
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    st.session_state["submit"] = False

# Cache data with a longer TTL to save memory
@st.cache_resource(ttl="2h")
def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower() if hasattr(uploaded_file, 'name') else uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    st.error(f"Formato de arquivo não suportado: {ext}")
    return None

# Main app configuration
st.set_page_config(page_title="LangChain: Chat com DataFrame Pandas", page_icon="🦜")
st.title("🦜 LangChain: Chat com DataFrame Pandas")

# Sidebar inputs
st.sidebar.title("Inputs necessários para o chat funcionar")

openai_api_key = st.sidebar.text_input(
    "Chave API OpenAI", 
    type="password", 
    help="A chave API é uma credencial fornecida pela OpenAI para acessar os modelos de IA, como o GPT-3.5."
)

uploaded_file = st.sidebar.file_uploader(
    "Envie um arquivo de dados", type=list(file_formats.keys()), help="Formatos suportados: CSV, XLS, XLSX", on_change=clear_submit
)

# Chat history
if "messages" not in st.session_state or st.sidebar.button("Limpar histórico de conversas"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso te ajudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# If file is uploaded, process the dataframe
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        prompt = st.chat_input(placeholder="Pergunte sobre os dados")
        if prompt:
            enhanced_prompt = f"{prompt} (Sempre forneça informações detalhadas sobre os dados e em português.)"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            if not openai_api_key:
                st.info("Por favor, insira sua chave API OpenAI para continuar.")
                st.stop()

            # Set up the LLM and agent
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
            pandas_df_agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True, agent_type=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent({"input": enhanced_prompt}, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response['output']})
                st.write(response['output'])
                
                # If the response contains a plot query, execute and display it
                if 'query' in response:
                    exec(response['query'])  # Execute the plot code

                    # Display the plot using st.pyplot
                    st.pyplot(plt)
