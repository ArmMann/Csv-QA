#!/usr/bin/env python3
"""
CSV RAG Pipeline - Chat interface for querying tabular data.

Usage:
    python main.py                    # Interactive mode
    python main.py "your question"    # Single question mode
    python main.py --verbose          # Interactive mode with debug output
"""

import os
import sys
import argparse
from dotenv import load_dotenv


load_dotenv()

from src.pipeline import RAGPipeline


def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("  CSV RAG Pipeline - Chat Over Tabular Data")
    print("=" * 60)
    print("Ask questions about clients, invoices, and line items.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'help' for example questions.")
    print("-" * 60)


def print_help():
    """Print example questions."""
    print("\nExample questions you can ask:")
    print("  - List all clients with their industries.")
    print("  - Which clients are based in the UK?")
    print("  - List all invoices issued in March 2024 with their statuses.")
    print("  - Which invoices are currently marked as 'Overdue'?")
    print("  - For each service_name in InvoiceLineItems, how many line items are there?")
    print("  - List all invoices for Acme Corp with their invoice IDs, dates, and statuses.")
    print("  - For invoice I1001, list all line items with service name, quantity, unit price, tax rate.")
    print("  - Which client has the highest total billed amount in 2024?")
    print()


def interactive_mode(pipeline: RAGPipeline):
    """Run the interactive chat interface."""
    print_banner()

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if question.lower() == 'help':
            print_help()
            continue

        # Get answer from pipeline
        result = pipeline.ask(question)

        print(f"\nA: {result.answer}")

        # Show debug info on failure or if verbose
        if not result.success:
            print(f"\n[Debug] Code:\n{result.generated_code}")
            print(f"[Debug] Error: {result.error}")
        elif pipeline.verbose and result.generated_code:
            print(f"\n[Debug] Code:\n{result.generated_code}")


def single_question_mode(pipeline: RAGPipeline, question: str):
    """Answer a single question and exit."""
    result = pipeline.ask(question)
    print(result.answer)

    # Show debug info on failure or if verbose
    if not result.success:
        print(f"\n[Debug] Code:\n{result.generated_code}")
        print(f"[Debug] Error: {result.error}")
    elif pipeline.verbose:
        print(f"\n[Debug] Generated code:\n{result.generated_code}")
        if result.formatted_result:
            print(f"\n[Debug] Raw result:\n{result.formatted_result}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chat interface for querying tabular data using RAG."
    )
    parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='Single question to answer (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose/debug output'
    )
    parser.add_argument(
        '-d', '--data-dir',
        default='data',
        help='Directory containing the Excel data files (default: data)'
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it:")
        print("  export GROQ_API_KEY=your_api_key_here")
        sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = RAGPipeline(
            data_dir=args.data_dir,
            verbose=args.verbose
        )
        pipeline.load()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    # Run in appropriate mode
    if args.question:
        single_question_mode(pipeline, args.question)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
