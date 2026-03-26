import pandas as pd
import json
import os
import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We need the local model configuration to run the AI
from src.utils.model_registry import get_model_object
dspy.settings.configure(lm=get_model_object("gpt-4.1-mini"))

def extract_metadata(df):
    """
    This simulates what the backend does when it first loads your CSV.
    Instead of sending the data, it extracts the schema and summary.
    """
    print("\n--- STEP 1: EXTRACTING METADATA FROM CSV ---")
    
    # 1. Get column names and data types
    columns_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 2. Get basic numerical statistics (mean, min, max, missing values)
    summary_stats = df.describe().to_dict()
    
    # 3. Get missing value counts
    missing_values = df.isnull().sum().to_dict()
    
    metadata = {
        "columns": columns_info,
        "sample_statistics": summary_stats,
        "missing_values": missing_values,
        "total_rows": len(df)
    }
    
    # This is the actual JSON string that gets sent to the AI!
    metadata_json = json.dumps(metadata, indent=2)
    print("Here is the exact (small) metadata string that will be sent to the AI:\n")
    print(metadata_json[:500] + "...\n(Truncated for readability)")
    
    return metadata_json

def generate_python_code(metadata_json, user_query):
    """
    This simulates the AI agent taking the metadata and a user query, 
    and outputting executable Python code to answer the query.
    """
    print(f"\n--- STEP 2: SENDING METADATA + QUERY TO AI ---")
    print(f"User Query: '{user_query}'")
    
    # A simplified version of the system prompt the backend uses
    prompt = f"""
    You are an AI Data Scientist. Your goal is to write a Python script using pandas to answer the user's query.
    
    Here is the exact schema and metadata of the user's dataframe (named 'df'):
    {metadata_json}
    
    User Query: {user_query}
    
    Rules:
    1. Assume the dataframe is already loaded in a variable called `df`.
    2. Write ONLY standard Python code. Do not wrap it in markdown block quotes (like ```python).
    3. Save the final answer/result to a variable called `final_result`.
    """
    
    print("\nRequesting code from OpenAI (gpt-4.1-mini)...")
    try:
        lm = get_model_object("gpt-4.1-mini")
        response = lm(prompt)
        # Assuming DSPy returns a list of generations
        code = response[0] if isinstance(response, list) else response
        
        # Clean up Markdown backticks if the model ignored the rule
        code = code.replace("```python", "").replace("```", "").strip()
        
        print("\n--- STEP 3: RECEIVED PYTHON CODE FROM AI ---")
        print(code)
        return code
    except Exception as e:
        print(f"\nError: {e}")
        print("Please make sure your .env file has a valid OPENAI_API_KEY.")
        return None

def execute_generated_code(df, generated_code):
    """
    This simulates the backend securely executing the AI's Python code locally.
    """
    print("\n--- STEP 4: EXECUTING GENERATED CODE LOCALLY ---")
    
    # Create an isolated environment dictionary
    # We pass the real 'df' object into this isolated environment
    local_env = {
        "df": df,
        "pd": pd,
    }
    
    try:
        # Run the AI-generated code inside our environment
        exec(generated_code, {}, local_env)
        
        # Extract the final result variable the AI created
        answer = local_env.get("final_result", "Code executed successfully, but 'final_result' was not defined.")
        
        print("\n--- FINAL OUTPUT ---")
        print("The execution engine returned this answer:")
        print(answer)
        
    except Exception as e:
        print(f"\nThe generated code had an error during execution: {e}")

if __name__ == "__main__":
    # Test file path
    csv_path = "Housing.csv"
    
    if not os.path.exists(csv_path):
        print(f"Please place a valid CSV file named '{csv_path}' in this directory.")
    else:
        # Load the dataframe
        print(f"Loading '{csv_path}' into memory...")
        df = pd.read_csv(csv_path)
        
        # 1. Extract context
        metadata_context = extract_metadata(df)
        
        # 2. Ask a question and get code
        query = "What is the average price of houses with exactly 3 bedrooms?"
        code = generate_python_code(metadata_context, query)
        
        # 3. Execute the code locally
        if code:
            execute_generated_code(df, code)
