# 📊 Sheets AI Pro with RAG

Sheets AI Pro is a powerful Streamlit web application that allows you to interact with your Google Sheets data using natural language. Powered by Google's Gemini large language model and a Retrieval-Augmented Generation (RAG) system, this tool can understand your questions, perform data queries, create visualizations, and even update your spreadsheet—all through a simple chat interface.

The RAG system enhances the AI's understanding of your specific dataset by creating semantic chunks of your data, enabling more accurate and context-aware responses.

## ✨ Key Features
- **Natural Language Interaction**: Chat with your data to ask questions, get summaries, and perform updates.  
- **Retrieval-Augmented Generation (RAG)**: A sophisticated RAG system provides deep, context-aware understanding of your data for more accurate AI responses.  
- Supports both basic text matching and advanced semantic search (using sentence-transformers and FAISS).  
- **AI-Powered Data Analysis**: Automatically generate pandas code to query and filter your data based on your questions.  
- **Intelligent Updates**: The AI can infer your intent to update the sheet and apply changes to the correct rows and columns.  
- **Dynamic Visualizations**: Ask the AI to create interactive charts and graphs (bar charts, line charts, pie charts, etc.) using Plotly Express.  
- **Data Explorer**: View, sort, and download your entire Google Sheet dataset directly within the app.  
- **RAG Insights**: Explore how the RAG system has chunked your data and test its semantic search capabilities.  
- **Secure and Configurable**: Uses service account credentials for secure access to Google Sheets and environment variables for API keys.  

## 🛠️ Tech Stack
- **Backend**: Python  
- **Frontend**: Streamlit  
- **AI Model**: Google Gemini Pro  
- **Data Backend**: Google Sheets  
- **RAG/Embeddings**: Sentence-Transformers, FAISS  
- **Data Manipulation**: Pandas  

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
- Python 3.8+  
- A Google Cloud Platform (GCP) project  
- A Google Sheet you want to connect to  

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up Google Cloud Credentials
**Enable APIs**: In your GCP project, make sure the Google Drive API and Google Sheets API are enabled.  

**Create a Service Account**:
1. Go to "IAM & Admin" > "Service Accounts" in your GCP console.  
2. Click "Create Service Account".  
3. Give it a name (e.g., `sheets-ai-pro-service-account`) and grant it the **Editor** role.  
4. After creating the account, click on it → "Keys" tab → "Add Key" → "Create new key" → select JSON.  
5. A JSON file will be downloaded. Rename this file to `credentials.json` and place it in the project root.  

**Share your Google Sheet**:  
- Open your Google Sheet and share it with the `client_email` from your `credentials.json`.  
- Grant it **Editor** permissions.  

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root with:
```env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
GOOGLE_SHEET_URL="YOUR_GOOGLE_SHEET_URL"
GEMINI_MODEL="gemini-2.5-flash"
```

- **GEMINI_API_KEY**: Your Google Gemini API key.  
- **GOOGLE_SHEET_URL**: The full URL of the Google Sheet to connect.  
- **GEMINI_MODEL**: (Optional) Defaults to `gemini-2.5-flash`.  

## 🚀 Running the Application
```bash
streamlit run app.py
```
The app will open in your browser, connect to Google Sheets, build the RAG index, and let you interact with your data!  

## 📂 Project Structure
```
.
├── app.py                  # Main Streamlit application file
├── update_util.py          # Utility functions for advanced sheet operations
├── requirements.txt        # Python package dependencies
├── credentials.json        # GCP service account credentials (DO NOT COMMIT)
├── .env                    # Environment variables (DO NOT COMMIT)
└── README.md               # Project documentation
```

## 💡 How It Works
1. **Data Loading**: The app securely connects to Google Sheets via service account credentials, loads the data into Pandas.  
2. **RAG Indexing**: The `AdvancedRAGSystem` (or `SimpleRAGSystem`) processes the DataFrame, creates chunks (row, column, summary), and stores embeddings in a FAISS index.  
3. **User Interaction**: You ask a question in the chat.  
4. **Intent Classification**: The AI (with RAG context) classifies your intent as QUERY, UPDATE, or VISUALIZE.  
5. **Code Generation**: For queries/visuals, AI generates a Pandas or Plotly command.  
6. **Execution & Display**: The code runs, showing a table or chart in the chat.  
7. **Sheet Updates**: Updates are applied directly to Google Sheets via gspread.  

## 🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs and feature requests.  
