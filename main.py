import os
import time
from dotenv import load_dotenv
from typing import Dict, List
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import inspect, text

# Load env variables
load_dotenv()

# -----------------------------
# PostgreSQL connection
# -----------------------------
POSTGRES_URI = (
    f"postgresql+psycopg2://"
    f"{os.getenv('PSQL_ROOT_USER')}:"
    f"{os.getenv('PSQL_ROOT_PWD')}@"
    f"{os.getenv('PSQL_HOST')}/"
    f"{os.getenv('PSQL_DATABASE')}"
)

db = SQLDatabase.from_uri(
    POSTGRES_URI,
    sample_rows_in_table_info=20,
    include_tables=None,
    max_string_length=5000
)

# -----------------------------
# Simple Database Helper
# -----------------------------
class DatabaseHelper:
    def __init__(self, db_uri: str):
        self.db = SQLDatabase.from_uri(db_uri)
        self.engine = self.db._engine
        
    def get_all_tables(self) -> List[str]:
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def execute_sql_directly(self, query: str):
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()
                    return {
                        "success": True,
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows)
                    }
                else:
                    return {
                        "success": True,
                        "message": "Query executed successfully",
                        "rowcount": result.rowcount
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

# -----------------------------
# Helper function to extract text from Gemini response
# -----------------------------
def extract_text_from_gemini_response(response):
    """Extract text content from Gemini response object"""
    # If response is a string, return it
    if isinstance(response, str):
        return response.strip()
    
    # If response has a content attribute
    if hasattr(response, 'content'):
        content = response.content
        # Handle different content types
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            # Extract text from list of parts
            texts = []
            for part in content:
                if hasattr(part, 'text'):
                    texts.append(part.text)
                elif isinstance(part, dict) and 'text' in part:
                    texts.append(part['text'])
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(texts).strip()
    
    # If response is a dictionary
    elif isinstance(response, dict):
        if 'content' in response:
            content = response['content']
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        texts.append(part['text'])
                    elif isinstance(part, str):
                        texts.append(part)
                return " ".join(texts).strip()
        elif 'text' in response:
            return response['text'].strip()
    
    # Try to convert to string as last resort
    return str(response).strip()

# -----------------------------
# Enhanced Query Handler
# -----------------------------
def process_natural_language_query(question: str):
    """
    Handle natural language query to SQL with explanation
    """
    helper = DatabaseHelper(POSTGRES_URI)
    
    print("\n" + "="*70)
    print(f"PROCESSING: '{question}'")
    print("="*70)
    
    try:
        # Initialize Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=os.getenv("GENAI_API_KEY"),
        )
        
        # Get database info for context
        tables = helper.get_all_tables()
        schema_info = f"Available tables: {', '.join(tables[:10])}" if len(tables) > 10 else f"Available tables: {', '.join(tables)}"
        
        # Step 1: Generate SQL query
        print("\n[1/3] 🔍 Generating SQL query...")
        
        # Clear prompt for SQL generation
        sql_prompt = f"""You are a PostgreSQL expert. Convert the user's question into a valid SQL query.

Database context: {schema_info}

User question: {question}

Generate ONLY the SQL query. Do not include any explanations, markdown, or additional text.
Just the SQL query. If you cannot generate a query, respond with: ERROR: Cannot generate SQL

SQL Query:"""
        
        # Get response from Gemini
        sql_response = llm.invoke(sql_prompt)
        
        # Extract SQL using helper function
        generated_sql = extract_text_from_gemini_response(sql_response)
        
        # Clean the SQL
        generated_sql = generated_sql.strip()
        
        # Remove markdown code blocks
        generated_sql = generated_sql.replace("```sql", "").replace("```", "")
        generated_sql = generated_sql.strip()
        
        # Remove any trailing semicolons if they cause issues
        if generated_sql.endswith(";"):
            generated_sql = generated_sql[:-1]
        generated_sql = generated_sql.strip()
        
        # Check for error
        if generated_sql.startswith("ERROR:") or "cannot generate" in generated_sql.lower():
            print(f"\n❌ Could not generate SQL for this question.")
            print(f"\n💡 Try asking questions like:")
            print("   - 'Show me all users'")
            print("   - 'Count the number of orders'")
            print("   - 'What are the table names?'")
            return None
        
        print(f"\n✅ Generated SQL:")
        print("-"*60)
        print(generated_sql)
        print("-"*60)
        
        # Step 2: Execute SQL
        print("\n[2/3] ⚡ Executing SQL query...")
        
        # Add semicolon back for execution
        exec_sql = generated_sql + ";" if not generated_sql.endswith(";") else generated_sql
        query_result = helper.execute_sql_directly(exec_sql)
        
        if not query_result.get("success", False):
            print(f"\n❌ SQL Execution Error: {query_result.get('error', 'Unknown error')}")
            return None
        
        # Step 3: Generate explanation
        print("\n[3/3] 💡 Generating explanation...")
        
        # Prepare result summary for explanation
        result_summary = ""
        if "rows" in query_result and query_result["rows"]:
            row_count = query_result["row_count"]
            result_summary = f"The query returned {row_count} row{'s' if row_count != 1 else ''}."
            
            if row_count > 0:
                columns = query_result["columns"]
                result_summary += f"\nColumns: {', '.join(columns)}"
                
                # Add sample results
                if row_count <= 5:
                    result_summary += "\n\nAll results:"
                    for i, row in enumerate(query_result["rows"], 1):
                        row_dict = dict(zip(columns, row))
                        # Format values nicely
                        formatted_values = []
                        for col, val in zip(columns, row):
                            if val is None:
                                formatted_values.append(f"{col}: NULL")
                            else:
                                formatted_values.append(f"{col}: {val}")
                        result_summary += f"\n{i}. " + ", ".join(formatted_values)
                else:
                    result_summary += f"\n\nFirst 3 of {row_count} results:"
                    for i, row in enumerate(query_result["rows"][:3], 1):
                        row_dict = dict(zip(columns, row))
                        formatted_values = []
                        for col, val in zip(columns[:3], row[:3]):  # Show first 3 columns
                            if val is None:
                                formatted_values.append(f"{col}: NULL")
                            else:
                                formatted_values.append(f"{col}: {val}")
                        result_summary += f"\n{i}. " + ", ".join(formatted_values)
                        if len(columns) > 3:
                            result_summary += f" ... (+{len(columns)-3} more columns)"
        else:
            result_summary = f"✓ {query_result.get('message', 'Query executed successfully')}"
        
        # Generate explanation
        explanation_prompt = f"""Explain the following in simple, natural language:

Question: {question}

SQL Query Used: {generated_sql}

Query Results: {result_summary}

Provide a clear explanation in plain English. Just give the explanation, no markdown, no code blocks, no formatting.

Explanation:"""
        
        explanation_response = llm.invoke(explanation_prompt)
        explanation = extract_text_from_gemini_response(explanation_response)
        
        # Clean explanation - remove any markdown formatting
        explanation = explanation.replace("**", "").replace("__", "").replace("#", "").strip()
        
        # Display final results
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        print(f"\n❓ Question: {question}")
        
        print(f"\n🔧 SQL Query Used:")
        print("-"*40)
        print(generated_sql)
        print("-"*40)
        
        print(f"\n📈 Query Results:")
        print("-"*40)
        
        if "rows" in query_result and query_result["rows"]:
            row_count = query_result["row_count"]
            print(f"📊 Total rows: {row_count}")
            
            if row_count > 0:
                columns = query_result["columns"]
                print(f"📋 Columns: {', '.join(columns)}")
                
                print(f"\n📄 Data (showing first {min(3, row_count)} rows):")
                print("-"*30)
                
                # Simple display
                for i, row in enumerate(query_result["rows"][:3], 1):
                    row_values = []
                    for val in row:
                        if val is None:
                            row_values.append("NULL")
                        else:
                            row_values.append(str(val))
                    print(f"{i}. {' | '.join(row_values)}")
                    
                if row_count > 3:
                    print(f"   ... and {row_count - 3} more rows")
        else:
            print(f"✓ {query_result.get('message', 'Success')}")
        
        print("-"*40)
        
        print(f"\n💡 Explanation:")
        print("-"*70)
        print(explanation)
        print("-"*70)
        
        return {
            "question": question,
            "sql_query": generated_sql,
            "results": query_result,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# Main Chat Interface
# -----------------------------
def main():
    print("\n" + "="*70)
    print("🤖 AI DATABASE ASSISTANT")
    print("="*70)
    print("Ask questions in natural language about your database.")
    print("I'll convert them to SQL, run them, and explain the results.")
    print("\nExamples:")
    print("  • 'Show me all users'")
    print("  • 'How many orders do we have?'")
    print("  • 'What tables are in the database?'")
    print("  • 'List the top 5 customers by sales'")
    print("  • 'Find inactive users from last month'")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        question = input("\n💬 Your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\n👋 Goodbye! Have a great day!")
            break
        
        if not question:
            continue
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        
        # Process the query
        result = process_natural_language_query(question)
        
        # Ask if user wants to continue
        if result:
            continue_choice = input("\n↩️  Ask another question? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                print("\n👋 Goodbye!")
                break
        else:
            print("\n⚠️  Let's try another question or rephrase it.")
            continue

if __name__ == "__main__":
    main()