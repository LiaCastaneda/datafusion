#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Compare query results between two benchmark runs for correctness verification.

This script compares the actual query output (not timing) between two runs
to detect correctness regressions.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Couldn't import modules -- run `./bench.sh venv` first")
    raise


def read_result_file(file_path: Path) -> Optional[List[str]]:
    """Read a query result file and return lines (skipping header)."""
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header line, return data lines
    return [line.strip() for line in lines[1:] if line.strip()]


def normalize_line(line: str) -> str:
    """Normalize a result line for comparison (trim whitespace from each field)."""
    # Split by pipe, strip each field, rejoin
    fields = [field.strip() for field in line.split('|')]
    return '|'.join(fields)


def compare_results(
    baseline_dir: Path,
    comparison_dir: Path,
) -> None:
    """Compare query results between two directories."""
    console = Console()

    # use basename as the column names
    baseline_header = baseline_dir.parent.stem
    comparison_header = comparison_dir.parent.stem

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query", style="dim", width=12)
    table.add_column(baseline_header, justify="right", style="dim")
    table.add_column(comparison_header, justify="right", style="dim")
    table.add_column("Status", justify="right", style="dim")

    matching_count = 0
    different_count = 0
    missing_count = 0
    different_queries = []

    # Find all .out files in baseline directory
    result_files = sorted(baseline_dir.glob("q*.out"))

    for baseline_file in result_files:
        query_name = baseline_file.stem  # e.g., "q21"
        query_num = query_name[1:]  # e.g., "21"
        comparison_file = comparison_dir / baseline_file.name

        # Read results
        baseline_lines = read_result_file(baseline_file)
        comparison_lines = read_result_file(comparison_file)

        # Check if files exist
        if baseline_lines is None:
            table.add_row(
                f"Q{query_num}",
                "MISSING",
                "?" if comparison_lines is None else f"{len(comparison_lines)} rows",
                "no baseline"
            )
            missing_count += 1
            continue

        if comparison_lines is None:
            table.add_row(
                f"Q{query_num}",
                f"{len(baseline_lines)} rows",
                "MISSING",
                "no comparison"
            )
            missing_count += 1
            continue

        # Normalize and compare
        baseline_normalized = [normalize_line(line) for line in baseline_lines]
        comparison_normalized = [normalize_line(line) for line in comparison_lines]

        if baseline_normalized == comparison_normalized:
            table.add_row(
                f"Q{query_num}",
                f"{len(baseline_lines)} rows",
                f"{len(comparison_lines)} rows",
                "✓ MATCH"
            )
            matching_count += 1
        else:
            table.add_row(
                f"Q{query_num}",
                f"{len(baseline_lines)} rows",
                f"{len(comparison_lines)} rows",
                "✗ DIFFERENT"
            )
            different_count += 1
            different_queries.append(query_num)

    console.print(table)

    # Summary table
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Result Comparison Summary", justify="left", style="dim")
    summary_table.add_column("", justify="right", style="dim")

    summary_table.add_row("Queries Matching", str(matching_count))
    summary_table.add_row("Queries Different", str(different_count))
    summary_table.add_row("Queries Missing", str(missing_count))

    console.print(summary_table)

    # Show which queries differ
    if different_queries:
        console.print(f"\n[bold red]Queries with different results: {', '.join(f'Q{q}' for q in different_queries)}[/bold red]")
        console.print("[yellow]This indicates a correctness regression.[/yellow]")
        console.print("[dim]Run with --detailed to see differences[/dim]\n")


def print_detailed_diff(baseline_file: Path, comparison_file: Path, num_lines: int = 20):
    """Print detailed differences between two result files."""
    baseline_lines = read_result_file(baseline_file)
    comparison_lines = read_result_file(comparison_file)

    if baseline_lines is None or comparison_lines is None:
        return

    console = Console()

    console.print(f"  [bold]Row counts:[/bold]")
    console.print(f"    Baseline:   {len(baseline_lines)} rows")
    console.print(f"    Comparison: {len(comparison_lines)} rows")

    if len(baseline_lines) != len(comparison_lines):
        console.print(f"  [bold red]⚠ Row count mismatch![/bold red]")

    console.print(f"\n  [bold]First {num_lines} rows:[/bold]")
    max_rows = max(len(baseline_lines), len(comparison_lines))
    lines_to_show = min(num_lines, max_rows)

    for i in range(lines_to_show):
        baseline_val = baseline_lines[i] if i < len(baseline_lines) else "[red]<missing>[/red]"
        comparison_val = comparison_lines[i] if i < len(comparison_lines) else "[red]<missing>[/red]"

        if normalize_line(str(baseline_val)) != normalize_line(str(comparison_val)):
            console.print(f"  [bold yellow]Row {i+1}: DIFFERENT[/bold yellow]")
            console.print(f"    [dim]Baseline:  [/dim] {baseline_val}")
            console.print(f"    [dim]Comparison:[/dim] {comparison_val}")
        else:
            console.print(f"  [dim]Row {i+1}: {baseline_val}[/dim]")

    if max_rows > num_lines:
        console.print(f"\n  [dim]... and {max_rows - num_lines} more rows[/dim]")


def main() -> None:
    parser = ArgumentParser(
        description="Compare query results between two benchmark runs for correctness"
    )
    parser.add_argument(
        "baseline_path",
        type=Path,
        help="Directory containing baseline query results (e.g., results/branch1/results/)"
    )
    parser.add_argument(
        "comparison_path",
        type=Path,
        help="Directory containing comparison query results (e.g., results/branch2/results/)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed diff for queries with different results"
    )

    args = parser.parse_args()

    # Validate directories
    if not args.baseline_path.exists():
        print(f"Error: Baseline directory '{args.baseline_path}' does not exist")
        sys.exit(1)

    if not args.comparison_path.exists():
        print(f"Error: Comparison directory '{args.comparison_path}' does not exist")
        sys.exit(1)

    # Compare results
    compare_results(args.baseline_path, args.comparison_path)

    # Show detailed diff if requested
    if args.detailed:
        console = Console()
        console.print("\n[bold]Detailed Differences:[/bold]\n")

        result_files = sorted(args.baseline_path.glob("q*.out"))
        for baseline_file in result_files:
            query_name = baseline_file.stem
            query_num = query_name[1:]
            comparison_file = args.comparison_path / baseline_file.name

            baseline_lines = read_result_file(baseline_file)
            comparison_lines = read_result_file(comparison_file)

            if baseline_lines is None or comparison_lines is None:
                continue

            baseline_normalized = [normalize_line(line) for line in baseline_lines]
            comparison_normalized = [normalize_line(line) for line in comparison_lines]

            if baseline_normalized != comparison_normalized:
                console.print(f"[bold red]Query {query_num}:[/bold red]")
                print_detailed_diff(baseline_file, comparison_file)
                console.print()

    # Exit with appropriate code
    # Count different queries
    different_count = 0
    result_files = sorted(args.baseline_path.glob("q*.out"))
    for baseline_file in result_files:
        comparison_file = args.comparison_path / baseline_file.name
        baseline_lines = read_result_file(baseline_file)
        comparison_lines = read_result_file(comparison_file)

        if baseline_lines and comparison_lines:
            baseline_normalized = [normalize_line(line) for line in baseline_lines]
            comparison_normalized = [normalize_line(line) for line in comparison_lines]
            if baseline_normalized != comparison_normalized:
                different_count += 1

    if different_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
