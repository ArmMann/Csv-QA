# CSV RAG Pipeline - Solution Documentation

pipeline_best.log contains the best results with llama17b and oss120b.

A RAG-style question-answering system for querying tabular business data using natural language.

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Get a free API key from [Groq Console](https://console.groq.com/) and set it up:

```bash
# Option 1: Create .env file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Option 2: Export directly
export GROQ_API_KEY=your_api_key_here
```

### 3. Run the Application

```bash
# Interactive chat mode
python main.py

# Single question mode
python main.py "Which clients are based in the UK?"

# Run all test questions with full logging
python run_tests.py --log
```

## Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Question  │────▶│   Stage 1: LLM   │────▶│   Validate &    │
│                 │     │  (Code Generate) │     │   Execute Code  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Stage 2: LLM   │◀─────────────┘
                        │ (Answer Generate)│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Final Answer    │
                        └──────────────────┘
```

### Components

1. **Data Loader** (`src/data_loader.py`)

   - Loads Excel, parses date columns
   - Generates schema descriptions for LLM prompts
2. **Code Generator** (`src/code_generator.py`)

   - Stage 1 LLM: Converts natural language request to Pandas code
   - Uses `llm` model for accurate code generation
   - Includes schema context and sample data in prompts
   - Supports error feedback for automatic code correction
3. **Executor** (`src/executor.py`)

   - Validates generated code for safety (blocks dangerous patterns)
   - Executes code in a sandboxed environment
   - Formats results for human readability
4. **Answer Generator** (`src/answer_generator.py`)

   - Stage 2 LLM: Converts query results to natural language
5. **Pipeline** (`src/pipeline.py`)

   - Orchestrates all stages
   - Implements retry logic with error feedback

### Model Selection

| Stage             | Model                                                                                                                                          | Rationale                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Code Generation   | `llama-3.3-70b-versatile, openai/gpt-oss-120b`                                                                                               | bigger , errors cost more here                 |
| Answer Generation | `llama-3.1-8b-instant, openai/gpt-oss-20b, openai/gpt-oss-120b qwen/qwen3-32b(the outputs of this one should be cleared for <think> traces)` | to explore generation patterns with diff sizes |

## Hallucination Mitigation Strategies

1. **Code-Based Retrieval**: Instead of letting the LLM generate answers directly, it generates Pandas code that executes against actual data.
2. **Schema Context**: The LLM receives exact column names and data types, reducing incorrect field references.
3. **Sample Data**: Including sample rows to help the LLM understand  formats.
4. **Error Feedback Loop**: If code fails, the error is fed back to the LLM for self-correction (up to 2 retries).
5. **Answer Grounding**: Stage 2 LLM is explicitly instructed to only use data from the query result.
6. **Code Validation**: Blocks dangerous patterns like imports, file operations, and eval/exec calls.

## Assumptions

1. **Data Fits in Memory**.
2. **Schema Stability**.
3. **Date Format**: Dates are parseable by pandas.to_datetime().
4. Total = quantity × unit_price × (1 + tax_rate)
5. fx_rate_to_usd can be used for USD conversion if needed.
6. For the "European clients" question, assumes common European country names.

## Test Results

The results can be tracked with pipeline logs from test runs with different model combinations. Each log contains full execution traces: prompts, generated code, execution results, and final answers. **pipeline_best.log** contains the best results.

| Log File                        | Stage 1 (Code Gen) | Stage 2 (Answer Gen)                       |
| ------------------------------- | ------------------ | ------------------------------------------ |
| `pipeline_best.log`           | gpt-oss-120b       | llama-4-maverick-17b                       |
| `pipeline_llama70_llama8.log` | llama-3.3-70b      | llama-3.1-8b                               |
| `pipeline_oss120_qwen3.log()` | gpt-oss-120b       | qwen3-32b(answer contains thinking traces) |
| `pipeline_oss120_oss20.log`   | gpt-oss-120b       | gpt-oss-20b                                |
| `pipeline_oss120_oss120.log`  | gpt-oss-120b       | gpt-oss-120b                               |
| `pipeline_oss20_llama12.log`  | gpt-oss-20b        | llama-3.2-12b                              |

Run `python run_tests.py --log` to generate your own test run.
