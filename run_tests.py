#!/usr/bin/env python3
"""
Run all example questions and generate results.

Usage:
    python run_tests.py           # Run tests
    python run_tests.py --log     # Also save full output to pipeline.log
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from src.pipeline import RAGPipeline

load_dotenv()

EXAMPLE_QUESTIONS = [
    "List all clients with their industries.",
    "Which clients are based in the UK?",
    "List all invoices issued in March 2024 with their statuses.",
    "Which invoices are currently marked as 'Overdue'?",
    "For each service_name in InvoiceLineItems, how many line items are there?",
    "List all invoices for Acme Corp with their invoice IDs, invoice dates, due dates, and statuses.",
    "Show all invoices issued to Bright Legal in February 2024, including their status and currency.",
    "For invoice I1001, list all line items with service name, quantity, unit price, tax rate, and compute the line total (including tax) for each.",
    "For each client, compute the total amount billed in 2024 (including tax) across all their invoices.",
    "Which client has the highest total billed amount in 2024, and what is that total?",
    "Across all clients, which three services generated the most revenue in 2024? Show the total revenue per service.",
    "Which invoices are overdue as of 2024-12-31? List invoice ID, client name, invoice_date, due_date, and status.",
    "Group revenue by client country: for each country, compute the total billed amount in 2024 (including tax).",
    "For the service 'Contract Review', list all clients who purchased it and the total amount they paid for that service (including tax).",
    "Considering only European clients, what are the top 3 services by total revenue (including tax) in H2 2024 (2024-07-01 to 2024-12-31)?",
]


class Logger:
    """Simple logger that writes to both console and file."""
    def __init__(self, log_file=None):
        self.log_file = open(log_file, 'w') if log_file else None

    def write(self, msg=""):
        print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


def run_tests(log_file=None):
    """Run all example questions."""
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set. Please set it in .env file.")
        return

    log = Logger(log_file)
    log.write(f"Pipeline Test Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.write("=" * 80)

    pipeline = RAGPipeline(data_dir="data", verbose=False)
    pipeline.load()
    log.write("Pipeline ready.\n")

    results = []

    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        log.write(f"\n{'='*80}")
        log.write(f"[{i}/{len(EXAMPLE_QUESTIONS)}] {question}")
        log.write("=" * 80)

        result = pipeline.ask(question)

        # Log Stage 1 prompt
        if result.stage1_prompt:
            log.write("\n--- STAGE 1 PROMPT ---")
            log.write(result.stage1_prompt)

        # Log generated code
        if result.generated_code:
            log.write("\n--- STAGE 1 GENERATED CODE ---")
            log.write(result.generated_code)

        # Log execution result
        if result.formatted_result:
            log.write("\n--- EXECUTION RESULT ---")
            log.write(result.formatted_result)
        elif result.error:
            log.write(f"\n--- ERROR ---")
            log.write(result.error)

        # Log Stage 2 prompt
        if result.stage2_prompt:
            log.write("\n--- STAGE 2 PROMPT ---")
            log.write(result.stage2_prompt)

        # Log final answer
        log.write("\n--- FINAL ANSWER ---")
        log.write(result.answer)
        log.write(f"\nStatus: {'SUCCESS' if result.success else 'FAILED'}")

        results.append((question, result.answer, result.success))

    # Summary
    log.write("\n" + "=" * 80)
    log.write("SUMMARY")
    log.write("=" * 80)
    successful = sum(1 for _, _, s in results if s)
    log.write(f"Success: {successful}/{len(results)} questions")

    # Save markdown table
    markdown = "| Question | Answer |\n|----------|--------|\n"
    for question, answer, _ in results:
        clean_answer = answer.replace("|", "\\|").replace("\n", " ")
        markdown += f"| {question} | {clean_answer} |\n"

    with open("TEST_RESULTS.md", "w") as f:
        f.write("# Test Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(markdown)

    log.write("\nResults saved to TEST_RESULTS.md")
    if log_file:
        log.write(f"Full log saved to {log_file}")

    log.close()


if __name__ == "__main__":
    log_file = "pipeline.log" if "--log" in sys.argv else None
    run_tests(log_file)
