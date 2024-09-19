import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
from io import BytesIO

# Configuração do Plotly e Streamlit
pio.kaleido.scope.default_format = "png"
pio.templates.default = "plotly_white"

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    st.session_state["submit"] = False

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower() if hasattr(uploaded_file, 'name') else uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    st.error(f"Formato de arquivo não suportado: {ext}")
    return None

st.set_page_config(page_title="LangChain: Chat com DataFrame Pandas", page_icon="🦜")
st.title("🦜 LangChain: Chat com DataFrame Pandas")

uploaded_file = st.file_uploader(
    "Envie um arquivo de dados", type=list(file_formats.keys()), help="Formatos suportados: CSV, XLS, XLSX", on_change=clear_submit
)

openai_api_key = st.sidebar.text_input("Chave API OpenAI", type="password")

if "messages" not in st.session_state or st.sidebar.button("Limpar histórico de conversas"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso te ajudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        prompt = st.chat_input(placeholder="Pergunte sobre os dados")
        if prompt:
            # Adiciona instruções ao prompt, sem exibir na interface
            enhanced_prompt = f"{prompt} (Sempre sugira um gráfico do Plotly.)"
            st.session_state.messages.append({"role": "user", "content": prompt})  # Apenas o prompt original
            st.chat_message("user").write(prompt)

            if not openai_api_key:
                st.info("Por favor, insira sua chave API OpenAI para continuar.")
                st.stop()

            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
            pandas_df_agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, allow_dangerous_code=True, agent_type=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent({"input": enhanced_prompt}, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response['output']})
                st.write(response['output'])

                # Geração do gráfico com base no prompt
                chart_map = {
                    "histograma": px.histogram,
                    "linha": px.line,
                    "barra": px.bar,
                    "distribuição": px.scatter,
                    "pizza": px.pie,
                }

                # Verifica o tipo de gráfico solicitado
                for chart_name, chart_fn in chart_map.items():
                    if chart_name in enhanced_prompt.lower():
                        # Identifica a coluna a ser utilizada no gráfico
                        column = next((col for col in df.columns if col.lower() in enhanced_prompt.lower()), df.columns[0])

                        # Criação e exibição do gráfico
                        fig = chart_fn(df, x=column, title=f"{chart_name.title()} de {column}")
                        st.plotly_chart(fig)

                        # Salva o gráfico como PNG
                        img_bytes = BytesIO()
                        fig.write_image(img_bytes, format='png')
                        img_bytes.seek(0)

                        # Botão para download do gráfico em PNG
                        st.download_button(label="Baixar gráfico como PNG", data=img_bytes, file_name=f"{chart_name}_{column}.png", mime="image/png")
                        break
