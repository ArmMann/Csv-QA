"""Code validation and safe execution module."""

import ast
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional


# Allowed builtins for safe execution
SAFE_BUILTINS = {
    'len': len,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'list': list,
    'dict': dict,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'tuple': tuple,
    'set': set,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'True': True,
    'False': False,
    'None': None,
}

# Blocked patterns that indicate unsafe code (excluding safe imports)
BLOCKED_PATTERNS = [
    '__import__',
    'exec(',
    'eval(',
    'open(',
    'os.',
    'sys.',
    'subprocess',
    'file(',
    'input(',
    '__builtins__',
    '__class__',
    '__bases__',
    '__subclasses__',
    'getattr',
    'setattr',
    'delattr',
    'globals',
    'locals',
    'compile',
]

# Safe imports that we allow and pre-provide
SAFE_IMPORTS = ['datetime', 'timedelta', 'pd', 'np', 'pandas', 'numpy']


def strip_imports(code: str) -> str:
    """
    Remove import statements from code since we pre-provide common modules.
    """
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip import lines for safe modules
        if stripped.startswith('import ') or stripped.startswith('from '):
            # Check if it's a safe import we can skip
            is_safe = any(safe in stripped for safe in SAFE_IMPORTS)
            if is_safe:
                continue  # Skip this line, we pre-provide these
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def validate_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate generated code for safety and syntax.

    """
    # Check for blocked patterns
    code_lower = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"Unsafe pattern detected: {pattern}"

    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, None


def execute_code(
    code: str,
    dataframes: Dict[str, pd.DataFrame]
) -> Tuple[Any, Optional[str]]:
    """
    Execute validated code in a restricted environment.

    """
    # Strip safe imports 
    code = strip_imports(code)

    # Validate the cleaned code
    is_valid, error = validate_code(code)
    if not is_valid:
        return None, error

    # Create restricted execution environment with datetime support
    exec_globals = {
        '__builtins__': SAFE_BUILTINS,
        'pd': pd,
        'np': np,
        'datetime': datetime,
        'timedelta': timedelta,
    }

    # Add dataframes to the environment
    exec_globals.update(dataframes)

    # Execute the code
    exec_locals = {}
    try:
        # Parse AST to understand code structure
        tree = ast.parse(code)

        if not tree.body:
            return None, "Empty code"

        last_stmt = tree.body[-1]

        # Check if last statement is an assignment (handles multi-line assignments)
        if isinstance(last_stmt, ast.Assign):
            # Execute entire code, return the assigned variable
            exec(code, exec_globals, exec_locals)
            exec_globals.update(exec_locals)
            # Get the variable name from the AST
            if last_stmt.targets and isinstance(last_stmt.targets[0], ast.Name):
                var_name = last_stmt.targets[0].id
                result = exec_locals.get(var_name) or exec_globals.get(var_name)
            else:
                result = None
        elif isinstance(last_stmt, ast.Expr):
            # Last statement is an expression - execute setup, eval last
            if len(tree.body) > 1:
                # Execute all statements except the last
                setup_code = ast.unparse(ast.Module(body=tree.body[:-1], type_ignores=[]))
                exec(setup_code, exec_globals, exec_locals)
                exec_globals.update(exec_locals)
            # Evaluate the last expression
            last_expr = ast.unparse(last_stmt)
            result = eval(last_expr, exec_globals, exec_locals)
        else:
            # Other statement types - just execute all
            exec(code, exec_globals, exec_locals)
            result = None

        return result, None

    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"


def format_result(result: Any, max_rows: int = 20) -> str:
    """
    Format execution result for display and LLM consumption.

    """
    if result is None:
        return "No result"

    if isinstance(result, pd.DataFrame):
        if len(result) > max_rows:
            summary = f"DataFrame with {len(result)} rows (showing first {max_rows}):\n"
            return summary + result.head(max_rows).to_string(index=False)
        return f"DataFrame with {len(result)} rows:\n" + result.to_string(index=False)

    if isinstance(result, pd.Series):
        if len(result) > max_rows:
            summary = f"Series with {len(result)} items (showing first {max_rows}):\n"
            return summary + result.head(max_rows).to_string()
        return f"Series with {len(result)} items:\n" + result.to_string()

    if isinstance(result, (list, tuple)) and len(result) > 0:
        if isinstance(result[0], (pd.DataFrame, pd.Series)):
            return "\n\n".join(format_result(r, max_rows) for r in result)
        return str(result)

    if isinstance(result, float):
        # Format floats nicely
        if result == int(result):
            return str(int(result))
        return f"{result:.2f}"

    return str(result)
