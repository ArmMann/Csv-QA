"""Stage 2: Generate natural answers from query results."""

import os
from typing import Tuple, Union
from openai import OpenAI

from .code_generator import GROQ_API_BASE, get_client


ANSWER_GEN_MODEL = os.getenv("ANSWER_GEN_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")#"openai/gpt-oss-20b")#"qwen/qwen3-32b")#"llama-3.1-8b-instant")


def build_answer_prompt(question: str, result_summary: str) -> str:
    """ Stage 2 prompt for answer generation."""
    return f"""You are a professional data analyst assistant. Based on the data query result below, provide a clear and concise answer to the user's question.

    ## User Question:
    {question}

    ## Query Result:
    {result_summary}

    ## Instructions:
    1. Answer the question directly using the data provided
    2. Include specific numbers/values from the result
    3. Keep the answer concise but complete
    4. introduce any additional table formatting if strictly necessary for the answer.
    5. If the result is a table, summarize the key findings
    6. Do NOT make up any numbers - only use what's in the result
    7. Format currency values with appropriate symbols when relevant

    Answer:"""


def generate_answer(
    question: str,
    result_summary: str,
    generated_code: str = None,
    client: OpenAI = None,
    return_prompt: bool = False
) -> Union[str, Tuple[str, str]]:
    """
    Generate a natural language answer from query results.

    """
    if client is None:
        client = get_client()

    prompt = build_answer_prompt(question, result_summary)

    response = client.chat.completions.create(
        model=ANSWER_GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful data analyst. Provide clear, accurate answers based solely on the data provided. Never invent or hallucinate numbers."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()

    if return_prompt:
        return answer, prompt
    return answer


def generate_error_response(
    question: str,
    error_message: str,
    client: OpenAI = None
) -> str:
    """
    Generate a user-friendly error response.
    
    """
    if client is None:
        client = get_client()

    prompt = f"""The user asked a question about their business data, but we couldn't process it.

    ## User Question:
    {question}

    ## Technical Error:
    {error_message}

    ## Instructions:
    Provide a brief, friendly response explaining that we couldn't answer the question.
    If possible, suggest what might have gone wrong or how to rephrase the question.
    Keep it concise (2-3 sentences max).

    Response:"""

    response = client.chat.completions.create(
        model=ANSWER_GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant explaining data query issues to users."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()
