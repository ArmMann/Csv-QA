"""Data ingestion module for loading Excel files into Pandas DataFrames."""

import pandas as pd
from pathlib import Path
from typing import Dict


def load_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all Excel files from the data directory into DataFrames.

    Returns:
        Dictionary with DataFrame names as keys and DataFrames as values.
    """
    data_path = Path(data_dir)

    # Load  Excel 
    clients_df = pd.read_excel(data_path / "Clients.xlsx")
    invoices_df = pd.read_excel(data_path / "Invoices.xlsx")
    line_items_df = pd.read_excel(data_path / "InvoiceLineItems.xlsx")


    invoices_df['invoice_date'] = pd.to_datetime(invoices_df['invoice_date'])
    invoices_df['due_date'] = pd.to_datetime(invoices_df['due_date'])

    return {
        'clients_df': clients_df,
        'invoices_df': invoices_df,
        'line_items_df': line_items_df
    }


def get_schema_description(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a schema description string for the LLM prompt.

    """
    schema_parts = []

    for name, df in dataframes.items():
        cols_info = []
        for col in df.columns:
            dtype = df[col].dtype
            # Simplify dtype names for the LLM
            if 'datetime' in str(dtype):
                dtype_str = 'datetime'
            elif dtype == 'object':
                dtype_str = 'string'
            elif 'int' in str(dtype):
                dtype_str = 'int'
            elif 'float' in str(dtype):
                dtype_str = 'float'
            else:
                dtype_str = str(dtype)
            cols_info.append(f"{col} ({dtype_str})")

        schema_parts.append(f"- {name}: [{', '.join(cols_info)}]")

    return "\n".join(schema_parts)


def get_sample_data(dataframes: Dict[str, pd.DataFrame], n_rows: int = 2) -> str:
    """
    Get sample data from each DataFrame for context.

    Args:
        dataframes: Dictionary of DataFrames
        n_rows: Number of sample rows to include

    Returns:
        Formatted string with sample data.
    """
    samples = []
    for name, df in dataframes.items():
        sample = df.head(n_rows).to_string(index=False)
        samples.append(f"{name} sample:\n{sample}")

    return "\n\n".join(samples)
