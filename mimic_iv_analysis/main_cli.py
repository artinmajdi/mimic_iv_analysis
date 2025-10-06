#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main command-line interface for MIMIC-IV Analysis.
This module provides the main entry point with subcommands.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog='mimic',
        description='MIMIC-IV Analysis Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
                    mimic ui                             # Start the Streamlit dashboard
                    mimic convert --table_name pharmacy  # Convert pharmacy table to Parquet
                    mimic convert --table_name study     # Convert all tables to Parquet
                    mimic convert --help                 # Show convert command help
                    mimic merge                          # Merge all tables into a single Parquet file
                    mimic merge --path /path/to/file      # Merge all tables into a single Parquet file at the specified path
                    mimic merge --help                    # Show merge command help """
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )

    # UI subcommand
    ui_parser = subparsers.add_parser(
        'ui',
        help='Start the Streamlit dashboard',
        description='Launch the interactive MIMIC-IV analysis dashboard'
    )

    # Convert subcommand
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert MIMIC-IV tables to Parquet format',
        description='Convert MIMIC-IV CSV tables to optimized Parquet format'
    )


    # Merge subcommand
    merge_parser = subparsers.add_parser(
        'merge',
        help='Merge all tables into a single Parquet file',
        description='Merge all tables into a single Parquet file'
    )

    # Import TableNames here to avoid circular imports
    from mimic_iv_analysis.configurations import TableNames

    convert_parser.add_argument(
        '--table_name',
        type=str,
        choices=['study'] + [e.value for e in TableNames],
        required=True,
        help='Name of the table to convert'
    )

    merge_parser.add_argument(
        '--path',
        type=str,
        required=False,
        default=None,
        help='Path to the target Parquet file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle subcommands
    if args.command == 'ui':
        from mimic_iv_analysis.cli import run_dashboard
        run_dashboard()
    elif args.command == 'convert':
        from mimic_iv_analysis.io.data_loader import ParquetConverter
        # Set up sys.argv for the converter function
        sys.argv = ['convert', '--table_name', args.table_name]
        ParquetConverter.example_save_to_parquet(table_name=args.table_name)
    elif args.command == 'merge':
        from mimic_iv_analysis.io.data_loader import DataLoader
        DataLoader.example_export_merge_table(target_path=args.path if args.path else None)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()