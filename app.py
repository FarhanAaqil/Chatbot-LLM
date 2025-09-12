# Google gemini streamlit app to query, update, and visualize Google Sheets with natural language
# Enhanced with RAG (Retrieval-Augmented Generation) capabilities
# streamlit_ui.py

import os
import json
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import streamlit as st
import update_util
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError
import google.generativeai as genai
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from typing import List, Dict, Tuple
import pickle
import hashlib
import re
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False
    st.warning("⚠️ Advanced RAG features unavailable. Install 'sentence-transformers' and 'faiss-cpu' for full functionality.")

# ------------------ 1. Setup and Configuration ------------------ #

st.set_page_config(page_title="Sheets AI Pro with RAG", layout="wide", page_icon="📊")
load_dotenv()

# --- Helper Functions ---
def clear_chat():
    st.session_state.messages = []
    st.session_state.rag_clicked = False

@st.cache_data
def to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

# --- Secrets and API Configuration ---
try:
    creds_json = { "type": st.secrets["gcp_service_account"]["type"], "project_id": st.secrets["gcp_service_account"]["project_id"], "private_key_id": st.secrets["gcp_service_account"]["private_key_id"], "private_key": st.secrets["gcp_service_account"]["private_key"], "client_email": st.secrets["gcp_service_account"]["client_email"], "client_id": st.secrets["gcp_service_account"]["client_id"], "auth_uri": st.secrets["gcp_service_account"]["auth_uri"], "token_uri": st.secrets["gcp_service_account"]["token_uri"], "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"], "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"], "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]}
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    GOOGLE_SHEET_URL = st.secrets.get("GOOGLE_SHEET_URL")
    GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
except (KeyError, AttributeError, StreamlitSecretNotFoundError):
    creds_json = "credentials.json"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Default model to use if the configured one fails ---
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY or not GOOGLE_SHEET_URL:
    st.error("❌ Critical configuration missing. Please ensure GEMINI_API_KEY and GOOGLE_SHEET_URL are set.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# ------------------ 2. RAG System Implementation ------------------ #

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    if ADVANCED_RAG_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

class SimpleRAGSystem:
    """Fallback RAG system using basic text matching when advanced libraries aren't available"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        
    def create_document_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """Create semantic chunks from the dataframe for RAG"""
        chunks = []
        
        # Create row-level chunks
        for idx, row in df.iterrows():
            row_text = f"Row {idx + 1}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            chunks.append({
                'text': row_text,
                'type': 'row',
                'row_index': idx,
                'metadata': dict(row)
            })
        
        # Create column-level chunks
        for col in df.columns:
            col_stats = self.get_column_insights(df, col)
            chunks.append({
                'text': f"Column {col}: {col_stats}",
                'type': 'column',
                'column': col,
                'metadata': {'column_name': col, 'stats': col_stats}
            })
        
        # Create summary chunk
        summary_text = f"Dataset summary: {df.shape[0]} rows, {df.shape[1]} columns. Columns: {', '.join(df.columns)}"
        chunks.append({
            'text': summary_text,
            'type': 'summary',
            'metadata': {'total_rows': df.shape[0], 'total_cols': df.shape[1]}
        })
        
        return chunks
    
    def get_column_insights(self, df: pd.DataFrame, col: str) -> str:
        """Generate insights for a specific column"""
        if df[col].dtype in ['int64', 'float64']:
            return f"Numeric column with mean {df[col].mean():.2f}, min {df[col].min()}, max {df[col].max()}"
        else:
            unique_count = df[col].nunique()
            top_values = df[col].value_counts().head(3).to_dict()
            return f"Text column with {unique_count} unique values. Top values: {top_values}"
    
    def build_index(self, df: pd.DataFrame):
        """Build simple text index from dataframe"""
        chunks = self.create_document_chunks(df)
        self.documents = [chunk['text'] for chunk in chunks]
        self.metadata = chunks
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search using basic text matching"""
        if not self.documents:
            return []
        
        query_words = set(re.findall(r'\w+', query.lower()))
        results = []
        
        for i, doc in enumerate(self.documents):
            doc_words = set(re.findall(r'\w+', doc.lower()))
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                score = overlap / len(query_words.union(doc_words))
                results.append({
                    'text': doc,
                    'score': score,
                    'metadata': self.metadata[i]
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

class AdvancedRAGSystem(SimpleRAGSystem):
    """Advanced RAG system using sentence transformers and FAISS"""
    
    def __init__(self):
        super().__init__()
        self.embedding_model = load_embedding_model()
        self.index = None
        
    def build_index(self, df: pd.DataFrame):
        """Build FAISS index from dataframe"""
        chunks = self.create_document_chunks(df)
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = texts
        self.metadata = chunks
        
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents using semantic similarity"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        return results

@st.cache_resource
def get_rag_system():
    if ADVANCED_RAG_AVAILABLE:
        return AdvancedRAGSystem()
    else:
        return SimpleRAGSystem()

# ------------------ 3. Backend AI and Data Functions ------------------ #

@st.cache_resource(ttl=300)
def get_gspread_client():
    try:
        if isinstance(creds_json, dict):
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, SCOPES)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Failed to authorize Google Sheets client: {e}")
        st.stop()

@st.cache_data(ttl=60)
def get_worksheet_and_dataframe():
    try:
        gc = get_gspread_client()
        sh = gc.open_by_url(GOOGLE_SHEET_URL)

        try:
            worksheet = sh.sheet1  # Always use the first sheet
        except gspread.exceptions.GSpreadException as e:
            st.error(f"❌ Could not open the Google Sheet. It might be empty or inaccessible. Error: {e}")
            st.stop()

        all_values = worksheet.get_all_values()
        if not all_values:
            return worksheet, pd.DataFrame()
        raw_headers = all_values[0]
        data_rows = all_values[1:]
        valid_indices = [i for i, h in enumerate(raw_headers) if h]
        clean_headers = [h for h in raw_headers if h]
        if len(set(clean_headers)) != len(clean_headers):
            st.error(f"❌ Error: Duplicate headers found in your Google Sheet.")
            st.stop()
        filtered_data = [[row[i] for i in valid_indices] for row in data_rows]
        df = pd.DataFrame(filtered_data, columns=clean_headers)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return worksheet, df
    except Exception as e:
        st.error(f"An error occurred while fetching the sheet: {e}")
        st.stop()

def get_enhanced_dataframe_context(df: pd.DataFrame, max_unique_values=20) -> str:
    if df.empty: return "The Google Sheet is empty."
    profile = [f"The sheet has {df.shape[0]} rows and {df.shape[1]} columns.", "\nHere is a detailed summary of each column:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_summary = f"- Column '{col}' (Type: {dtype}):"
        if df[col].isnull().all():
            col_summary += "\n  - This column is completely empty."
            profile.append(col_summary)
            continue
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            stats = f"  - Stats: mean={df[col].mean():.2f}, median={df[col].median():.2f}, min={df[col].min()}, max={df[col].max()}"
            col_summary += f"\n{stats}"
        else:
            unique_vals = df[col].dropna().unique()
            num_unique = len(unique_vals)
            if num_unique <= max_unique_values:
                col_summary += f"\n  - Contains {num_unique} unique values: {', '.join(map(str, unique_vals))}"
            else:
                col_summary += f"\n  - Contains {num_unique} unique values. Top 5 are: {', '.join(map(str, df[col].value_counts().nlargest(5).index))}"
        profile.append(col_summary)
    return "\n".join(profile)

def run_generative_model(prompt: str):
    """Tries the configured model, falls back to the default model on error."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response
    except Exception as e:
        st.warning(f"⚠️ Configured model '{GEMINI_MODEL}' failed with error: {e}. Falling back to default model.")
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response

def get_ai_intent_with_rag(question: str, df_schema: dict, rag_system):
    # Get relevant context from RAG
    relevant_docs = rag_system.search(question, k=3)
    rag_context = "\n".join([f"- {doc['text']}" for doc in relevant_docs])
    
    prompt = f"""
    You are an intelligent assistant with access to relevant data context. Your task is to classify the user's intent as QUERY, UPDATE, or VISUALIZE.
    
    Available columns: {list(df_schema.keys())}
    The most likely ID column is 'ID'.
    
    Relevant data context:
    {rag_context}
    
    Return a single, minified JSON object.
    JSON Structure: {{"intent": "QUERY|UPDATE|VISUALIZE", "row_id": "string|number|null", "id_column": "string|null", "column_to_update": "string|null", "new_value": "string|number|null"}}
    User Query: "{question}"
    """
    try:
        response = run_generative_model(prompt)
        return json.loads(response.text.strip())
    except Exception: 
        return {"intent": "QUERY"}

def generate_code_with_rag(question: str, context: str, task: str, rag_system) -> str:
    # Get relevant context from RAG
    relevant_docs = rag_system.search(question, k=5)
    rag_context = "\n".join([f"- {doc['text']} (Score: {doc['score']:.3f})" for doc in relevant_docs])
    
    if task == "QUERY": 
        instruction = "Return a single, executable line of pandas code."
        examples = """- User Question: "Show me the records for employee with ID 105"\n- Your Output: df[df['ID'] == '105']"""
    elif task == "VISUALIZE": 
        instruction = "Return a single, executable line of Plotly Express code that generates a figure object."
        examples = """- User Question: "Plot a bar chart of statuses"\n- Your Output: px.bar(df, x='Status')"""
    else: 
        return ""

    prompt = f"""
    You are a senior data analyst and Python expert with access to relevant data context. Your task is to convert a user's natural language question into a single line of Python code.
    
    **Data Context:** You are working with a pandas DataFrame named `df`. Here is a detailed profile of the data:
    ---
    {context}
    ---
    
    **Relevant Data Context from RAG:**
    {rag_context}
    ---
    
    **Instructions:**
    1. Analyze the user's question and the relevant context.
    2. Based on the data context and RAG results, formulate a single line of code to accomplish the task.
    3. {instruction}
    4. Return ONLY the raw Python code. Do not add explanations, comments, or markdown formatting.
    
    **Examples:**
    {examples}
    
    **User's Question:** "{question}"
    """
    try:
        response = run_generative_model(prompt)
        return response.text.strip().replace("`", "").replace("python", "")
    except Exception as e:
        st.error(f"An error occurred during code generation: {e}")
        return ""

def generate_summary_with_rag(context: str, rag_system) -> str:
    # Get summary-relevant context
    relevant_docs = rag_system.search("summary statistics overview insights", k=10)
    rag_context = "\n".join([f"- {doc['text']}" for doc in relevant_docs])
    
    prompt = f"""
    You are a helpful data analyst with access to detailed data insights. Based on the following data profile and relevant context, provide a comprehensive, business-focused summary.
    
    **Data Profile:**
    {context}
    
    **Detailed Insights:**
    {rag_context}
    
    Focus on key insights, patterns, and actionable information a business user would find valuable.
    Your Summary:
    """
    try:
        response = run_generative_model(prompt)
        return response.text
    except Exception: 
        return "Could not generate a summary."

def update_row(worksheet, row_id, id_column, column_to_update, new_value):
    """Update a specific row in the Google Sheet"""
    try:
        # Get all values to find the row
        all_values = worksheet.get_all_values()
        headers = all_values[0]
        
        # Find column indices
        try:
            id_col_idx = headers.index(id_column)
            update_col_idx = headers.index(column_to_update)
        except ValueError as e:
            return f"❌ Column not found: {e}"
        
        # Find the row with matching ID
        for i, row in enumerate(all_values[1:], start=2):  # Start from row 2 (1-indexed)
            if row[id_col_idx] == str(row_id):
                # Update the cell
                worksheet.update_cell(i, update_col_idx + 1, new_value)
                return f"✅ Successfully updated {column_to_update} to '{new_value}' for ID {row_id}"
        
        return f"❌ No row found with {id_column} = {row_id}"
    
    except Exception as e:
        return f"❌ Error updating row: {e}"

# ------------------ 4. STREAMLIT USER INTERFACE ------------------ #

st.title("📊 Sheets AI Pro with RAG")
st.markdown("Your intelligent assistant for querying, updating, and visualizing Google Sheets with advanced RAG capabilities.")

if not ADVANCED_RAG_AVAILABLE:
    with st.expander("📦 Install Advanced RAG Features", expanded=True):
        st.markdown("""
        **To enable advanced semantic search and RAG capabilities, install these packages:**
        
        \`\`\`bash
        pip install sentence-transformers faiss-cpu
        \`\`\`
        
        **Current Status:** Using basic text matching RAG (still functional but less sophisticated)
        """)

try:
    worksheet, df = get_worksheet_and_dataframe()
    df_context = get_enhanced_dataframe_context(df)
    rag_system = get_rag_system()
    
    if not df.empty:
        with st.spinner("Building RAG index for enhanced AI capabilities..."):
            chunk_count = rag_system.build_index(df)
        rag_type = "Advanced (Semantic)" if ADVANCED_RAG_AVAILABLE else "Basic (Text Matching)"
        st.success(f"✅ RAG system initialized ({rag_type}) with {chunk_count} semantic chunks")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
except Exception as e:
    st.error(f"Failed to load the application. Please check your configuration. Error: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuration & Help")
    st.info(f"🔗 **Connected Sheet:** [Open Sheet]({GOOGLE_SHEET_URL})")
    
    st.subheader("💡 RAG Prompt Suggestions")
    with st.expander("🚀 Quick RAG Prompts", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Data Summary", use_container_width=True):
                st.session_state.suggested_prompt = "Give me a comprehensive summary of this dataset with key insights"
            if st.button("🔍 Find Patterns", use_container_width=True):
                st.session_state.suggested_prompt = "What patterns and trends can you identify in this data?"
            if st.button("📈 Top Values", use_container_width=True):
                st.session_state.suggested_prompt = "Show me the top performing records and their characteristics"
        
        with col2:
            if st.button("⚠️ Data Issues", use_container_width=True):
                st.session_state.suggested_prompt = "Identify any data quality issues or anomalies"
            if st.button("📋 Column Analysis", use_container_width=True):
                st.session_state.suggested_prompt = "Analyze each column and provide detailed statistics"
            if st.button("🎯 Recommendations", use_container_width=True):
                st.session_state.suggested_prompt = "Based on this data, what actionable recommendations do you have?"
    
    st.subheader("🧠 RAG System Status")
    if rag_system.documents:
        rag_type = "Advanced (Semantic)" if ADVANCED_RAG_AVAILABLE else "Basic (Text Matching)"
        st.success(f"✅ Active ({rag_type}) with {len(rag_system.documents)} chunks")
        
        # RAG search test
        with st.expander("🔍 Test RAG Search", expanded=False):
            test_query = st.text_input("Test semantic search:")
            if test_query:
                results = rag_system.search(test_query, k=3)
                for i, result in enumerate(results):
                    st.write(f"**Result {i+1}** (Score: {result['score']:.3f})")
                    st.write(result['text'][:200] + "...")
    else:
        st.warning("⚠️ RAG system not initialized")
    
    with st.expander("💡 **How to Use**", expanded=False):
        st.markdown("""
        - **AI Assistant**: Ask questions with enhanced RAG context retrieval
        - **AI Visualizer**: Create charts with semantic understanding
        - **Data Explorer**: View raw data and RAG insights
        - **RAG Features**: Semantic search, context-aware responses, intelligent data understanding
        """)
    
    st.button("🗑️ Clear Chat History", on_click=clear_chat, use_container_width=True, type="primary")

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Total Rows", f"{df.shape[0]:,}")
mcol2.metric("Total Columns", f"{df.shape[1]:,}")
mcol3.metric("RAG Chunks", f"{len(rag_system.documents) if rag_system.documents else 0:,}")
st.divider()

tab_assistant, tab_visualizer, tab_explorer, tab_rag = st.tabs(["🤖 AI Assistant", "📈 AI Visualizer", "📊 Data Explorer", "🧠 RAG Insights"])

with tab_assistant:
    st.header("Chat with your Data (RAG-Enhanced)")
    
    if 'suggested_prompt' in st.session_state and st.session_state.suggested_prompt:
        st.info(f"💡 Suggested: {st.session_state.suggested_prompt}")
        if st.button("Use This Prompt"):
            prompt = st.session_state.suggested_prompt
            st.session_state.suggested_prompt = ""
            # Process the suggested prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): 
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Assistant is thinking with RAG context..."):
                    df_schema = {c: str(df[c].dtype) for c in df.columns}
                    
                    intent_data = get_ai_intent_with_rag(prompt, df_schema, rag_system)
                    intent = intent_data.get("intent", "QUERY")
                    response_content = None
                    
                    if intent == "UPDATE":
                        row_id, id_col, col, new_val = (intent_data.get("row_id"), intent_data.get("id_column"), intent_data.get("column_to_update"), intent_data.get("new_value"))
                        if not all([row_id, id_col, col, new_val]):
                            response_content = "⚠️ The AI could not determine all required fields for the update."
                        else:
                            response_content = update_row(worksheet, row_id, id_col, col, new_val)
                            st.cache_data.clear()
                    elif intent == "QUERY":
                        code = generate_code_with_rag(prompt, df_context, "QUERY", rag_system)
                        if code:
                            try: 
                                response_content = eval(code, {"df": df, "pd": pd, "np": np})
                            except Exception as e: 
                                response_content = f"⚠️ Error executing code: {e}"
                        else: 
                            response_content = "I'm sorry, I couldn't generate the right code for that request."
                    else: 
                        response_content = "That looks like a request for a chart. Please try it in the 'AI Visualizer' tab!"
                    
                    if isinstance(response_content, pd.DataFrame):
                        st.dataframe(response_content)
                        try:
                            csv = response_content.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download as CSV", 
                                data=csv, 
                                file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                mime="text/csv",
                                key=f"download_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Download error: {e}")
                    else: 
                        st.markdown(str(response_content))
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.rerun()
    
    if st.button("✨ Generate AI Summary with RAG"):
        with st.spinner("Analyzing data with RAG and generating enhanced summary..."):
            summary = generate_summary_with_rag(df_context, rag_system)
            st.session_state.messages.append({"role": "assistant", "content": summary})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"])
                try:
                    csv = message["content"].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download as CSV", 
                        data=csv, 
                        file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                        mime="text/csv",
                        key=f"download_msg_{hash(str(message['content'].values.tobytes()))}"
                    )
                except Exception as e:
                    st.error(f"Download preparation failed: {e}")
            elif hasattr(message["content"], 'to_html'):  # Plotly figure check
                 st.plotly_chart(message["content"])
            else:
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question with RAG-enhanced understanding..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Assistant is thinking with RAG context..."):
                df_schema = {c: str(df[c].dtype) for c in df.columns}
                
                intent_data = get_ai_intent_with_rag(prompt, df_schema, rag_system)
                intent = intent_data.get("intent", "QUERY")
                response_content = None
                
                if intent == "UPDATE":
                    row_id, id_col, col, new_val = (intent_data.get("row_id"), intent_data.get("id_column"), intent_data.get("column_to_update"), intent_data.get("new_value"))
                    if not all([row_id, id_col, col, new_val]):
                        response_content = "⚠️ The AI could not determine all required fields for the update."
                    else:
                        response_content = update_row(worksheet, row_id, id_col, col, new_val)
                        st.cache_data.clear()
                elif intent == "QUERY":
                    code = generate_code_with_rag(prompt, df_context, "QUERY", rag_system)
                    if code:
                        try: 
                            response_content = eval(code, {"df": df, "pd": pd, "np": np})
                        except Exception as e: 
                            response_content = f"⚠️ Error executing code: {e}"
                    else: 
                        response_content = "I'm sorry, I couldn't generate the right code for that request."
                else: 
                    response_content = "That looks like a request for a chart. Please try it in the 'AI Visualizer' tab!"
                
                if isinstance(response_content, pd.DataFrame):
                    st.dataframe(response_content)
                    try:
                        csv = response_content.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download as CSV", 
                            data=csv, 
                            file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                            mime="text/csv",
                            key=f"download_new_{len(st.session_state.messages)}"
                        )
                    except Exception as e:
                        st.error(f"Download preparation failed: {e}")
                else: 
                    st.markdown(str(response_content))
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

with tab_visualizer:
    st.header("Create Interactive Visualizations (RAG-Enhanced)")
    st.info("Ask the AI to create a chart with enhanced context understanding. For example: `Plot a bar chart showing the count for each status`")
    
    st.subheader("📊 Quick Visualization Prompts")
    viz_col1, viz_col2, viz_col3 = st.columns(3)
    
    with viz_col1:
        if st.button("📊 Bar Chart", use_container_width=True):
            st.session_state.viz_suggested = "Create a bar chart showing the distribution of the main categorical column"
        if st.button("📈 Line Chart", use_container_width=True):
            st.session_state.viz_suggested = "Create a line chart showing trends over time"
    
    with viz_col2:
        if st.button("🥧 Pie Chart", use_container_width=True):
            st.session_state.viz_suggested = "Create a pie chart showing the proportion of different categories"
        if st.button("📉 Scatter Plot", use_container_width=True):
            st.session_state.viz_suggested = "Create a scatter plot to show correlation between numeric columns"
    
    with viz_col3:
        if st.button("📊 Histogram", use_container_width=True):
            st.session_state.viz_suggested = "Create a histogram showing the distribution of numeric values"
        if st.button("🔥 Heatmap", use_container_width=True):
            st.session_state.viz_suggested = "Create a heatmap showing correlations between numeric columns"
    
    if 'viz_suggested' in st.session_state and st.session_state.viz_suggested:
        st.info(f"💡 Suggested: {st.session_state.viz_suggested}")
        if st.button("Use This Visualization Prompt"):
            viz_prompt = st.session_state.viz_suggested
            st.session_state.viz_suggested = ""
            
            with st.spinner("Generating visualization with RAG context..."):
                st.session_state.viz_code = generate_code_with_rag(viz_prompt, df_context, "VISUALIZE", rag_system)
                if st.session_state.viz_code:
                    try: 
                        st.session_state.viz_fig = eval(st.session_state.viz_code, {"df": df, "px": px})
                    except Exception as e:
                        st.error(f"⚠️ Error executing visualization code: {e}")
                        st.session_state.viz_fig = None
                else:
                     st.warning("Could not generate visualization code for that request.")
                     st.session_state.viz_fig = None
            st.rerun()
    
    if 'viz_code' not in st.session_state: 
        st.session_state.viz_code = ""
    if 'viz_fig' not in st.session_state: 
        st.session_state.viz_fig = None
    
    if viz_prompt := st.chat_input("What chart would you like to create?"):
        with st.spinner("Generating visualization with RAG context..."):
            st.session_state.viz_code = generate_code_with_rag(viz_prompt, df_context, "VISUALIZE", rag_system)
            if st.session_state.viz_code:
                try: 
                    st.session_state.viz_fig = eval(st.session_state.viz_code, {"df": df, "px": px})
                except Exception as e:
                    st.error(f"⚠️ Error executing visualization code: {e}")
                    st.session_state.viz_fig = None
            else:
                 st.warning("Could not generate visualization code for that request.")
                 st.session_state.viz_fig = None
    
    if st.session_state.viz_fig:
        st.plotly_chart(st.session_state.viz_fig, use_container_width=True)
        with st.expander("Show Generated Code"): 
            st.code(st.session_state.viz_code, language="python")

with tab_explorer:
    st.header("Explore Your Raw Data")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Dataset Overview ({df.shape[0]} rows × {df.shape[1]} columns)")
    with col2:
        try:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv_data,
                file_name=f"full_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Download preparation failed: {e}")
    
    st.dataframe(df, use_container_width=True)

with tab_rag:
    st.header("🧠 RAG System Insights")
    rag_type = "Advanced (Semantic)" if ADVANCED_RAG_AVAILABLE else "Basic (Text Matching)"
    st.markdown(f"Explore the semantic understanding and retrieval capabilities of the RAG system. **Current Mode:** {rag_type}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Semantic Search")
        search_query = st.text_input("Enter a search query:", placeholder="e.g., high values, recent entries, status analysis")
        search_k = st.slider("Number of results:", 1, 10, 5)
        
        if search_query:
            results = rag_system.search(search_query, k=search_k)
            st.write(f"Found {len(results)} relevant chunks:")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1} - Score: {result['score']:.3f}"):
                    st.write("**Text:**", result['text'])
                    st.write("**Type:**", result['metadata'].get('type', 'unknown'))
                    if result['metadata'].get('type') == 'row':
                        st.write("**Row Index:**", result['metadata'].get('row_index'))
                    elif result['metadata'].get('type') == 'column':
                        st.write("**Column:**", result['metadata'].get('column'))
    
    with col2:
        st.subheader("RAG System Statistics")
        if rag_system.documents:
            chunk_types = {}
            for chunk in rag_system.metadata:
                chunk_type = chunk.get('type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            st.write("**Chunk Distribution:**")
            for chunk_type, count in chunk_types.items():
                st.write(f"- {chunk_type.title()}: {count}")
            
            st.write(f"**Total Chunks:** {len(rag_system.documents)}")
            if ADVANCED_RAG_AVAILABLE:
                st.write(f"**Embedding Dimension:** 384")  # MiniLM dimension
                st.write(f"**Index Type:** FAISS IndexFlatIP")
            else:
                st.write(f"**Search Method:** Basic text matching")
                st.write(f"**Upgrade Available:** Install sentence-transformers for semantic search")
        else:
            st.warning("RAG system not initialized with data")
        
        if st.button("🔄 Rebuild RAG Index"):
            with st.spinner("Rebuilding RAG index..."):
                chunk_count = rag_system.build_index(df)
                st.success(f"✅ RAG index rebuilt with {chunk_count} chunks")
                st.rerun()
