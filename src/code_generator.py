"""Stage 1: Question to Pandas Code."""

import os
import re
from openai import OpenAI
from typing import Dict, Tuple, Union
import pandas as pd

from .data_loader import get_schema_description, get_sample_data

#  API configuration
GROQ_API_BASE = "https://api.groq.com/openai/v1"
# Model for code generation on Groq (can be overridden via environment)
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "openai/gpt-oss-120b")#"qwen/qwen3-32b")#"llama-3.3-70b-versatile")


def get_client() -> OpenAI:
    """Get OpenAI client configured for Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url=GROQ_API_BASE)


def build_code_generation_prompt(
    question: str,
    dataframes: Dict[str, pd.DataFrame],
    include_samples: bool = True
) -> str:
    """
    Build the prompt for code generation.

    """
    schema = get_schema_description(dataframes)

    prompt = f"""You are a Python data analyst. Write Pandas code to answer questions about business data.

    ## Available DataFrames:
    {schema}

    ## Key Relationships:
    - clients_df.client_id links to invoices_df.client_id
    - invoices_df.invoice_id links to line_items_df.invoice_id

    ## Important Notes:
    - invoice_date and due_date are datetime objects (use .dt accessor for year, month, etc.)
    - tax_rate is a decimal (e.g., 0.2 means 20%)
    - Line total with tax = quantity * unit_price * (1 + tax_rate)
    - Month reference: January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9, October=10, November=11, December=12
    - European countries in data: UK, Germany, Netherlands, Norway, Switzerland, France, Spain, Ireland, Portugal
    - To compare dates: use pd.Timestamp('2024-12-31') for date comparisons
    - datetime is already available (no import needed)
    - DO NOT use import statements - pd, np, datetime are pre-provided

    ## Examples:

    Q: List all clients from Germany
    Code: clients_df[clients_df['country'] == 'Germany'][['client_name', 'industry']]

    Q: Count invoices per client
    Code: invoices_df.merge(clients_df, on='client_id').groupby('client_name')['invoice_id'].count()

    Q: Top 2 clients by total billed amount (with tax)
    Code: line_items_df.assign(total=lambda x: x['quantity'] * x['unit_price'] * (1 + x['tax_rate'])).merge(invoices_df, on='invoice_id').merge(clients_df, on='client_id').groupby('client_name')['total'].sum().nlargest(2)
    """

    if include_samples:
        samples = get_sample_data(dataframes, n_rows=2)
        prompt += f"""
        ## Sample Data:
        {samples}
        """

    prompt += f"""
    ## Question:
    {question}

    ## Instructions:
    1. Use ONLY: clients_df, invoices_df, line_items_df
    2. Use exact column names from schema
    3. NO import statements (pd, np, datetime are available)
    4. NO print statements - the last line must be an expression that returns the answer
    5. Output ONLY Python code, no explanations or markdown

    Code:"""

    return prompt


def generate_pandas_code(
    question: str,
    dataframes: Dict[str, pd.DataFrame],
    client: OpenAI = None,
    max_retries: int = 2,
    return_prompt: bool = False
) -> Union[str, Tuple[str, str]]:
    """
    Generate Pandas code from a natural language question.

    Args:
        question: User's natural language question
        dataframes: Dictionary of available DataFrames
        client: OpenAI client (created if not provided)
        max_retries: Number of retries on failure
        return_prompt: If True, return (code, prompt) tuple

    Returns:
        Generated Python code string, or (code, prompt) tuple
    """
    if client is None:
        client = get_client()

    prompt = build_code_generation_prompt(question, dataframes)

    response = client.chat.completions.create(
        model=CODE_GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise Python code generator. Output only valid Python code, no explanations. Never use import statements."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1000
    )

    code = response.choices[0].message.content.strip()

    # Strip <think>...</think> tags from reasoning models (e.g., qwen3)
    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL).strip()
    # Handle incomplete <think> blocks (no closing tag - model truncated)
    if '<think>' in code:
        code = re.sub(r'<think>.*', '', code, flags=re.DOTALL).strip()

    # Clean up any markdown formatting that might have slipped through
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    # Replace literal \n with actual newlines
    code = code.replace('\\n', '\n')

    code = code.strip()

    if return_prompt:
        return code, prompt
    return code


def generate_code_with_error_feedback(
    question: str,
    dataframes: Dict[str, pd.DataFrame],
    previous_code: str,
    error_message: str,
    client: OpenAI = None
) -> str:
    """
    Generate corrected code given a previous error.

    """
    if client is None:
        client = get_client()

    schema = get_schema_description(dataframes)

    prompt = f"""You wrote code that produced an error. Fix it.

    ## Available DataFrames:
    {schema}

    ## Relationships:
    - clients_df.client_id -> invoices_df.client_id
    - invoices_df.invoice_id -> line_items_df.invoice_id

    ## Original Question:
    {question}

    ## Your Previous Code:
    {previous_code}

    ## Error:
    {error_message}

    ## Instructions:
    - Fix the code to answer the question correctly
    - NO import statements (pd, np, datetime are pre-provided)
    - Output ONLY the corrected Python code

    Corrected Code:"""

    response = client.chat.completions.create(
        model=CODE_GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise Python code generator. Output only valid Python code that fixes the error. Never use import statements."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1000
    )

    code = response.choices[0].message.content.strip()

    # Strip <think>...</think> tags from reasoning models
    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL).strip()
    if '<think>' in code:
        code = re.sub(r'<think>.*', '', code, flags=re.DOTALL).strip()

    # Clean up markdown formatting
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()
