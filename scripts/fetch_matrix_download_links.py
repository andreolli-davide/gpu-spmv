#!/usr/bin/env python3
"""
Fetch actual download links from SuiteSparse Matrix Collection
by parsing search results from sparse.tamu.edu

Questo script ricerca le matrici su sparse.tamu.edu e estrae i link di download reali.
"""

import urllib.request
import urllib.parse
import re
import sys
import csv
from html.parser import HTMLParser
from typing import List, Dict, Tuple

# Matrices to find
CITED_MATRICES = [
    "Protein", "FEM/Spheres", "FEM/Cantilever", "Wind Tunnel", "FEM/Harbor",
    "QCD", "FEM/Ship", "Economics", "Epidemiology", "FEM/Accelerator",
    "Circuit", "Webbase", "LP", "Circuit5M", "ASIC_320k", "ASIC_680k",
    "webbase-1M", "cnr-2000", "eu-2005", "in-2004",
    "thermomech_dK", "ldoor", "mip1", "dc2", "FullChip", "ins2",
    "Ga41As41H72", "Si41Ge41H72", "Dense"
]

SUITESPARSE_BASE = "https://sparse.tamu.edu/"


class MatrixHTMLParser(HTMLParser):
    """Parse matrix information from SuiteSparse HTML search results"""

    def __init__(self):
        super().__init__()
        self.matrices = []
        self.in_tbody = False
        self.in_tr = False
        self.current_row = {}
        self.current_cell = ""
        self.cell_index = 0
        self.skip_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.in_tbody = True
        elif tag == "tr" and self.in_tbody:
            self.in_tr = True
            self.current_row = {}
            self.cell_index = 0
        elif tag == "td" and self.in_tr:
            # Check for column classes
            attrs_dict = dict(attrs)
            self.skip_cell = "pagination" in attrs_dict.get("class", "")
            self.current_cell = ""
        elif tag == "a" and self.in_tr:
            # Extract href for group and name links
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href and not href.startswith("/?"):
                # This might be a matrix link like /LAW/webbase-2001
                parts = href.strip("/").split("/")
                if len(parts) == 2:
                    self.current_row["group"] = parts[0]
                    self.current_row["name"] = parts[1]

    def handle_endtag(self, tag):
        if tag == "tbody":
            self.in_tbody = False
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if "name" in self.current_row and "group" in self.current_row:
                self.matrices.append(self.current_row)
        elif tag == "td" and self.in_tr and not self.skip_cell:
            # Save cell content based on column index
            if self.cell_index == 1:
                self.current_row["id"] = self.current_cell.strip()
            elif self.cell_index == 3:
                self.current_row["rows"] = self.current_cell.strip()
            elif self.cell_index == 4:
                self.current_row["cols"] = self.current_cell.strip()
            elif self.cell_index == 5:
                self.current_row["nnz"] = self.current_cell.strip()
            self.cell_index += 1
            self.current_cell = ""

    def handle_data(self, data):
        if self.in_tr and not self.skip_cell:
            self.current_cell += data


def search_matrix(matrix_name: str) -> Dict:
    """
    Search for a matrix on SuiteSparse and return its download links.
    Returns dict with 'group', 'name', and download URLs or empty dict if not found.
    """
    # URL-encode the search query
    search_url = f"{SUITESPARSE_BASE}?filterrific%5Bsearch_query%5D={urllib.parse.quote(matrix_name)}&filterrific%5Bsorted_by%5D=nonzeros_desc"

    try:
        with urllib.request.urlopen(search_url, timeout=10) as response:
            html_content = response.read().decode('utf-8')

        # Parse HTML to extract matrix info
        parser = MatrixHTMLParser()
        parser.feed(html_content)

        # Find exact or partial match
        for matrix in parser.matrices:
            if matrix["name"].lower() == matrix_name.lower() or \
               matrix_name.lower() in matrix["name"].lower():
                return matrix

        # If no exact match found, return None
        return None

    except Exception as e:
        print(f"Error searching for {matrix_name}: {e}", file=sys.stderr)
        return None


def get_download_urls(group: str, name: str) -> Dict[str, str]:
    """Generate download URLs for a matrix"""
    base = "https://suitesparse-collection-website.herokuapp.com"
    return {
        "matlab": f"{base}/mat/{group}/{name}.mat",
        "rb": f"{base}/RB/{group}/{name}.tar.gz",
        "mm": f"{base}/MM/{group}/{name}.tar.gz",
    }


def fetch_all_matrices(matrices: List[str] = None) -> List[Dict]:
    """
    Fetch all matrices and their download links.
    Returns list of dicts with matrix info and URLs.
    """
    if matrices is None:
        matrices = CITED_MATRICES

    results = []
    print(f"Searching for {len(matrices)} matrices...", file=sys.stderr)

    for i, matrix in enumerate(matrices, 1):
        print(f"  [{i}/{len(matrices)}] Searching for: {matrix}...", file=sys.stderr, end="")
        found = search_matrix(matrix)

        if found:
            urls = get_download_urls(found["group"], found["name"])
            result = {
                "requested_name": matrix,
                "actual_name": found["name"],
                "group": found["group"],
                "rows": found.get("rows", "N/A"),
                "cols": found.get("cols", "N/A"),
                "nnz": found.get("nnz", "N/A"),
                "matrix_url": f"{SUITESPARSE_BASE}{found['group']}/{found['name']}",
                **urls
            }
            results.append(result)
            print(f" ✓ Found", file=sys.stderr)
        else:
            result = {
                "requested_name": matrix,
                "actual_name": "NOT FOUND",
                "group": "N/A",
                "rows": "N/A",
                "cols": "N/A",
                "nnz": "N/A",
                "matrix_url": "N/A",
                "matlab": "N/A",
                "rb": "N/A",
                "mm": "N/A"
            }
            results.append(result)
            print(f" ✗ Not found", file=sys.stderr)

    return results


def output_markdown(results: List[Dict], output_file: str = None):
    """Output in Markdown table format"""
    lines = [
        "# Sparse Matrix Download Links",
        "",
        "Complete list of download links for matrices cited in GPU SpMV papers.",
        "",
        "| Requested Name | Actual Name | Group | Rows | Cols | NNZ | Matrix Market | RB | MATLAB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for result in results:
        mm_link = f"[MM]({result['mm']})" if result['mm'] != "N/A" else "N/A"
        rb_link = f"[RB]({result['rb']})" if result['rb'] != "N/A" else "N/A"
        matlab_link = f"[MAT]({result['matlab']})" if result['matlab'] != "N/A" else "N/A"

        line = f"| {result['requested_name']} | {result['actual_name']} | {result['group']} | {result['rows']} | {result['cols']} | {result['nnz']} | {mm_link} | {rb_link} | {matlab_link} |"
        lines.append(line)

    output = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nMarkdown saved to: {output_file}", file=sys.stderr)
    else:
        print(output)


def output_csv(results: List[Dict], output_file: str = None):
    """Output in CSV format"""
    fieldnames = ['requested_name', 'actual_name', 'group', 'rows', 'cols', 'nnz', 'matrix_url', 'matlab', 'rb', 'mm']

    if output_file:
        f = open(output_file, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

    if output_file:
        f.close()
        print(f"\nCSV saved to: {output_file}", file=sys.stderr)


def output_bash_script(results: List[Dict], output_file: str = None, fmt: str = 'mm'):
    """Generate bash download script"""
    lines = [
        "#!/bin/bash",
        "# Auto-generated matrix download script",
        "# Downloaded from: https://sparse.tamu.edu/",
        "",
        "set -e",
        "DOWNLOAD_DIR='sparse_matrices'",
        "mkdir -p \"$DOWNLOAD_DIR\"",
        "cd \"$DOWNLOAD_DIR\"",
        "",
    ]

    # Find format code
    format_map = {'mm': 'MM', 'rb': 'RB', 'matlab': 'mat'}
    format_code = format_map.get(fmt, 'MM')
    ext_map = {'mm': '.tar.gz', 'rb': '.tar.gz', 'matlab': '.mat'}
    ext = ext_map.get(fmt, '.tar.gz')

    lines.extend([
        f"# Using {fmt.upper()} format",
        "",
        "# Matrices to download",
        "declare -A MATRICES",
    ])

    for result in results:
        if result['actual_name'] != "NOT FOUND":
            lines.append(f"MATRICES[\"{result['actual_name']}\"]='{result['group']}'")

    lines.extend([
        "",
        "# Download",
        f"BASE_URL='https://suitesparse-collection-website.herokuapp.com/{format_code}'",
        "",
        "echo \"Downloading ${#MATRICES[@]} matrices in {fmt} format...\"",
        "",
        "for name in \"${!MATRICES[@]}\"; do",
        "    group=\"${MATRICES[$name]}\"",
        "    url=\"$BASE_URL/$group/$name{ext}\"",
        "    echo \"Downloading: $name\"",
        "    if wget -q \"$url\" -O \"$name{ext}\"; then",
        "        echo \"  ✓ Success\"",
        "    else",
        "        echo \"  ✗ Failed: $url\"",
        "    fi",
        "done",
        "",
        "echo 'Done!'",
    ])

    output = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        import os
        os.chmod(output_file, 0o755)
        print(f"\nBash script saved to: {output_file}", file=sys.stderr)
    else:
        print(output)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch download links from SuiteSparse Matrix Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all matrices and output as markdown
  python3 fetch_matrix_download_links.py --format markdown --output links.md

  # Generate bash download script
  python3 fetch_matrix_download_links.py --format bash --output download.sh

  # Search for specific matrices
  python3 fetch_matrix_download_links.py webbase-1M ldoor circuit
        """
    )
    parser.add_argument(
        'matrices',
        nargs='*',
        help='Specific matrices to search (if empty, searches all cited matrices)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['markdown', 'csv', 'bash'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--matrix-format', '-m',
        choices=['mm', 'rb', 'matlab'],
        default='mm',
        help='Matrix download format (default: mm)'
    )

    args = parser.parse_args()

    # Determine which matrices to search
    if args.matrices:
        matrices_to_search = args.matrices
    else:
        matrices_to_search = CITED_MATRICES

    # Fetch all matrices
    results = fetch_all_matrices(matrices_to_search)

    # Output in requested format
    if args.format == 'markdown':
        output_markdown(results, args.output)
    elif args.format == 'csv':
        output_csv(results, args.output)
    elif args.format == 'bash':
        output_bash_script(results, args.output, args.matrix_format)

    # Print summary
    found = sum(1 for r in results if r['actual_name'] != "NOT FOUND")
    not_found = len(results) - found
    print(f"\nSummary: {found} found, {not_found} not found", file=sys.stderr)


if __name__ == '__main__':
    main()
