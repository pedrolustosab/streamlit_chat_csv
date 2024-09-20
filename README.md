# LangChain Chat with Pandas DataFrame

This project combines the power of Streamlit, LangChain, OpenAI, and Pandas to create an interactive chat interface for seamless data exploration. Users can upload datasets and interact with them using natural language queries. By leveraging LangChain's `create_pandas_dataframe_agent` function, the system interprets user queries and delivers insightful responses from the uploaded data, powered by OpenAI's language model.

## Features

- **Seamless Data Upload**: Upload datasets in various formats including CSV, XLS, XLSX, XLSM, and XLSB.
- **Natural Language Queries**: Ask questions about your data using natural language, powered by OpenAI's GPT-3.5-turbo model.
- **LangChain Integration**: Utilize LangChain's `create_pandas_dataframe_agent` for effective interaction with Pandas dataframes.
- **Improved Performance**: Enjoy faster response times with data caching using Streamlit's `@st.cache_data` decorator (cached for 2 hours).
- **Secure API Access**: Securely enter your OpenAI API key via a password-protected input field in the Streamlit sidebar.
- **Clear Conversation History**: Reset the chat at any time with a button to clear conversation history.
- **Real-Time Interaction**: Get instant responses to queries about your dataset directly in the chat interface.

## Requirements

- **Python**: 3.8+
- **Necessary Libraries**:
  - `streamlit`
  - `langchain`
  - `langchain_experimental`
  - `openai`
  - `pandas`

To install the required libraries, run:

```bash
pip install -r requirements.txt
