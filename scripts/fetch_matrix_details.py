#!/usr/bin/env python3
"""
Fetch detailed matrix information from SuiteSparse Collection
Parses matrix pages and extracts structure, symmetry, and other properties
"""

import urllib.request
import urllib.error
from html.parser import HTMLParser
import sys
import time
from typing import Dict, List, Tuple

# All 29 matrices with their group/name
MATRICES = [
    ("Williams", "pdb1HYS"),
    ("Williams", "consph"),
    ("Williams", "mac_econ_fwd500"),
    ("Williams", "mc2depi"),
    ("Williams", "cop20k_A"),
    ("Freescale", "circuit5M"),
    ("LAW", "webbase-2001"),
    ("Qaplib", "lp_nug20"),
    ("Sandia", "ASIC_320k"),
    ("Sandia", "ASIC_680k"),
    ("Williams", "webbase-1M"),
    ("LAW", "cnr-2000"),
    ("LAW", "eu-2005"),
    ("LAW", "in-2004"),
    ("Botonakis", "thermomech_dK"),
    ("GHS_psdef", "ldoor"),
    ("Andrianov", "mip1"),
    ("HVDC", "hvdc2"),
    ("DNVS", "fullb"),
    ("Andrianov", "ins2"),
    ("PARSEC", "Ga41As41H72"),
    ("PARSEC", "Si41Ge41H72"),
    ("DNVS", "shipsec5"),
    ("Oberwolfach", "windscreen"),
    ("Schmid", "thermal2"),
    ("Um", "offshore"),
    ("FEMLAB", "poisson3Db"),
    ("Janna", "Hook_1498"),
    ("Nasa", "nasasrb"),
]

SUITESPARSE_BASE = "https://sparse.tamu.edu/"


class MatrixInfoParser(HTMLParser):
    """Parse matrix information from SuiteSparse HTML page"""

    def __init__(self):
        super().__init__()
        self.in_tr = False
        self.in_th = False
        self.in_td = False
        self.current_key = ""
        self.current_value = ""
        self.info = {}

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_tr = True
            self.current_key = ""
            self.current_value = ""
        elif tag == "th" and self.in_tr:
            self.in_th = True
            self.current_key = ""
        elif tag == "td" and self.in_tr:
            self.in_td = True
            self.current_value = ""

    def handle_endtag(self, tag):
        if tag == "th" and self.in_th:
            self.in_th = False
        elif tag == "td" and self.in_td:
            self.in_td = False
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            # Store key-value pair
            if self.current_key and self.current_value:
                self.info[self.current_key.strip()] = self.current_value.strip()

    def handle_data(self, data):
        if self.in_th:
            self.current_key += data
        elif self.in_td:
            self.current_value += data


def fetch_matrix_info(group: str, name: str) -> Dict[str, str]:
    """
    Fetch detailed matrix information from SuiteSparse page
    Returns dict with matrix properties
    """
    url = f"{SUITESPARSE_BASE}{group}/{name}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            html_content = response.read().decode('utf-8', errors='ignore')

        # Parse HTML
        parser = MatrixInfoParser()
        parser.feed(html_content)

        return parser.info

    except urllib.error.URLError as e:
        print(f"  ✗ Network error: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"  ✗ Error: {e}", file=sys.stderr)
        return {}


def fetch_all_matrices() -> List[Tuple[str, str, Dict]]:
    """
    Fetch information for all matrices
    Returns list of (group, name, info_dict)
    """
    results = []

    print(f"Fetching information for {len(MATRICES)} matrices...\n", file=sys.stderr)

    for i, (group, name) in enumerate(MATRICES, 1):
        print(f"[{i:2d}/{len(MATRICES)}] {name:20s} ", end="", file=sys.stderr)
        sys.stderr.flush()

        info = fetch_matrix_info(group, name)

        if info:
            results.append((group, name, info))
            print("✓", file=sys.stderr)
        else:
            results.append((group, name, {}))
            print("✗ (no data)", file=sys.stderr)

        # Rate limiting
        time.sleep(0.5)

    return results


def format_markdown(results: List[Tuple[str, str, Dict]]) -> str:
    """
    Format results as markdown
    """
    lines = [
        "# Sparse Matrix Detailed Properties",
        "",
        "Complete information extracted from SuiteSparse Collection pages.",
        "",
        f"**Total matrices**: {len(results)}",
        "",
        "---",
        "",
    ]

    for i, (group, name, info) in enumerate(results, 1):
        lines.extend([
            f"## {i}. {name}",
            "",
            f"**Collection**: {group}  ",
            f"**URL**: https://sparse.tamu.edu/{group}/{name}",
            "",
        ])

        if info:
            lines.append("### Properties")
            lines.append("")

            for key, value in info.items():
                # Format key nicely
                formatted_key = key.replace("_", " ").title()
                lines.append(f"- **{formatted_key}**: {value}")

            lines.append("")
        else:
            lines.append("*(No detailed information available)*")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch detailed matrix information from SuiteSparse Collection',
    )
    parser.add_argument(
        '--output', '-o',
        help='Output markdown file (default: stdout)'
    )

    args = parser.parse_args()

    # Fetch all matrix information
    results = fetch_all_matrices()

    # Format as markdown
    markdown = format_markdown(results)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(markdown)
        print(f"\n✓ Markdown saved to: {args.output}", file=sys.stderr)
    else:
        print(markdown)

    # Summary
    found = sum(1 for _, _, info in results if info)
    print(f"\nSummary: {found} / {len(results)} matrices with detailed info", file=sys.stderr)


if __name__ == '__main__':
    main()
