# Auto Analyst Backend: Capabilities Overview

The Auto Analyst Backend is a robust Python/FastAPI-based application designed to provide AI-powered data analytics, machine learning, and reporting functionalities. It allows users to interact with multiple specialized AI agents to process, analyze, and visualize data seamlessly.

Here is a comprehensive breakdown of what you can do using this backend:

## 1. Interact with Specialized AI Data Agents
The core of the system is the AI agent architecture configured via DSPy and defined in `agents_config.json`. You can interact with specific agents via the `/chat/{agent_name}` endpoint to achieve distinct data tasks:

*   **Data Preprocessing Agent (`preprocessing_agent`)**: Automatically cleans and prepares datasets using Pandas and NumPy. It handles missing values, detects column types, and converts strings to datetime formats.
*   **Statistical Analytics Agent (`statistical_analytics_agent`)**: Performs robust statistical analysis (e.g., linear regression, seasonal decomposition) using `statsmodels`, including categorical encodings and significance testing.
*   **Data Visualization Agent (`data_viz_agent`)**: Generates interactive, high-quality, and visually appealing charts using `Plotly` based on specific styling instructions (e.g., handling charts for heatmaps, histograms, pie charts).
*   **Machine Learning Agent (`sk_learn_agent`)**: Uses `scikit-learn` to train, evaluate, and extract insights from machine learning models (classification, regression, clustering) along with feature importance insights.
*   **Feature Engineering Agent (`feature_engineering_agent`)**: An advanced tool to create interaction terms, apply scaling, handle temporal data, and perform dimensionality reduction.
*   **Polars Agent (`polars_agent`)**: Optimized for high-performance, large-scale data processing using the `Polars` library (leveraging lazy evaluation and streaming).

*Note: These agents can operate individually or collaboratively as part of a multi-agent "planner" pipeline.*

## 2. Generate "Deep Analysis" Reports
The backend provides a `/deep_analysis` route designed for extensive, multi-step analytical tasks:

*   **End-to-End Execution**: You can submit a high-level analytical goal, and the backend orchestrates a plan, executes the required Python code, and synthesizes the results.
*   **Interactive Outputs**: The deep analysis yields summaries, generated code, interactive Plotly visualizations, synthesis notes, and a final conclusion.
*   **HTML Report Generation**: It can automatically format the deep analysis results into a downloadable HTML report (`get_html_report`).
*   **History & Tracking**: Every deep analysis is stored in a database (tracking tokens used, credits consumed, errors, and execution time) allowing users to fetch their historical reports.

## 3. Session & Dataset Management
Through the `session_routes`:
*   **Stateful Chats**: The backend tracks user interactions over time, maintaining contextual memory of the chat history.
*   **Dataset Handling**: Users can upload datasets which are then cached and made accessible to the AI agents for querying.

## 4. Admin Analytics & Telemetry Dashboard
The `analytics_routes` provide a comprehensive administrative view into the backend's performance and economics:
*   **Cost & Token Tracking**: Monitor API usage across different LLM providers, calculating daily costs, token consumption, and projecting future expenses.
*   **User Activity**: Track Daily Active Users (DAU), new users, average session duration, and queries per session.
*   **Model Performance**: Evaluate the response times, success rates, and token outputs of different configured AI models.
*   **Real-time WebSockets**: Connect to `/analytics/dashboard/realtime` for live streaming updates of system usage.

## 5. Extensibility and Safety
*   **Code Execution Sandboxing**: The backend securely formats, executes, and returns results for generated Python code, preventing malicious operations.
*   **Dynamic Safeguards**: Configurable model parameters (like temperature and max tokens) are safely applied based on the session or system defaults.
*   **CORS & Environment Setup**: The backend is production-ready with strict origin verification, environment variable loading via `.env`, and database interactions via SQLAlchemy.

---
**Summary:** You can use this backend to build a full-fledged "AI Data Scientist" platform. A frontend application can leverage these APIs to let users upload CSVs, ask questions in plain English, and receive back clean datasets, interactive charts, ML models, or fully synthesized HTML analysis reports.

## How to Run Locally

If you are developing or running the Auto Analyst Backend on your local machine, follow these steps:

### 1. Set Up Your Environment
Ensure you have Python installed (Python 3.10+ recommended). Then, create a virtual environment, activate it, and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
A `.env` file containing the default environment variables has been generated for you in the root of the backend directory.

Make sure to edit `.env` and configure any necessary API keys (like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) if you plan on using those models.

### 3. Initialize the Database & Populate Templates
The backend uses SQLite locally by default (specified via `DATABASE_URL` in `.env`). Run these initialization scripts to set up your tables and load the agent configurations from `agents_config.json`:

```bash
# Initialize the database schema
python scripts/init_production_db.py

# Populate the AI Agent templates into the DB
python scripts/populate_agent_templates.py sync
```

### 4. Start the Application
Once the environment is prepped and the DB is ready, start the FastAPI server using `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

The backend API will now be accessible locally at [http://localhost:7860](http://localhost:7860). You can also view the interactive Swagger API documentation at [http://localhost:7860/docs](http://localhost:7860/docs).
