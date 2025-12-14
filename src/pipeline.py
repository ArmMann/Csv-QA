"""Main RAG pipeline orchestrating all stages."""

import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI

from .data_loader import load_data
from .code_generator import generate_pandas_code, generate_code_with_error_feedback, get_client, build_code_generation_prompt
from .executor import execute_code, format_result
from .answer_generator import generate_answer, generate_error_response, build_answer_prompt


@dataclass
class PipelineResult:
    """Result from the RAG pipeline."""
    question: str
    answer: str
    generated_code: Optional[str] = None
    raw_result: Optional[Any] = None
    formatted_result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    stage1_prompt: Optional[str] = None
    stage2_prompt: Optional[str] = None


class RAGPipeline:
    """
    RAG-style pipeline for answering questions about tabular data.

    Pipeline stages:
    1. NL-to-Pandas code generation 
    2. Code validation and execution
    3. Result-to-NL answer generation 
    """

    def __init__(
        self,
        data_dir: str = "data",
        max_retries: int = 2,
        verbose: bool = False
    ):
        self.data_dir = data_dir
        self.max_retries = max_retries
        self.verbose = verbose
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.client: Optional[OpenAI] = None

    def load(self) -> None:
        """Load data and initialize the LLM client."""
        if self.verbose:
            print("Loading data...")
        self.dataframes = load_data(self.data_dir)
        if self.verbose:
            for name, df in self.dataframes.items():
                print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
            print("Initializing LLM client...")
        self.client = get_client()

    def ask(self, question: str) -> PipelineResult:
        """Answer a natural language question about the data."""
        if not self.dataframes:
            self.load()

        # Build Stage 1 prompt for logging
        stage1_prompt = build_code_generation_prompt(question, self.dataframes)

        # Stage 1: Generate Pandas code
        if self.verbose:
            print(f"\n[Stage 1] Generating code for: {question}")

        try:
            code, _ = generate_pandas_code(
                question, self.dataframes, self.client, return_prompt=True
            )
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            return PipelineResult(
                question=question,
                answer=f"Sorry, I couldn't generate code. Error: {str(e)}",
                success=False,
                error=error_msg,
                stage1_prompt=stage1_prompt
            )

        if self.verbose:
            print(f"Generated code:\n{code}\n")

        # Execute code with retry loop
        result = None
        error = None

        for attempt in range(self.max_retries + 1):
            if self.verbose and attempt > 0:
                print(f"[Retry {attempt}] Attempting to fix code...")

            result, error = execute_code(code, self.dataframes)

            if error is None:
                break

            if self.verbose:
                print(f"Execution error: {error}")

            if attempt < self.max_retries:
                try:
                    code = generate_code_with_error_feedback(
                        question, self.dataframes, code, error, self.client
                    )
                    if self.verbose:
                        print(f"Fixed code:\n{code}\n")
                except Exception as e:
                    error = f"Code fix failed: {str(e)}"
                    break

        # If all retries failed
        if error is not None:
            if self.verbose:
                print(f"[Stage 1 Failed] {error}")
            try:
                answer = generate_error_response(question, error, self.client)
            except:
                answer = f"Sorry, I couldn't answer your question: {error}"

            return PipelineResult(
                question=question,
                answer=answer,
                generated_code=code,
                success=False,
                error=error,
                stage1_prompt=stage1_prompt
            )

        # Format the result
        formatted_result = format_result(result)
        if self.verbose:
            print(f"[Stage 1 Complete] Result:\n{formatted_result}\n")

        # Build Stage 2 prompt
        stage2_prompt = build_answer_prompt(question, formatted_result)

        # Stage 2: Generate natural language answer
        if self.verbose:
            print("[Stage 2] Generating answer...")

        try:
            answer = generate_answer(question, formatted_result, code, self.client)
        except Exception as e:
            answer = f"Here are the results:\n\n{formatted_result}"

        if self.verbose:
            print(f"[Stage 2 Complete] Answer: {answer}\n")

        return PipelineResult(
            question=question,
            answer=answer,
            generated_code=code,
            raw_result=result,
            formatted_result=formatted_result,
            success=True,
            stage1_prompt=stage1_prompt,
            stage2_prompt=stage2_prompt
        )

    def ask_batch(self, questions: list) -> list:
        """Answer multiple questions."""
        return [self.ask(q) for q in questions]
