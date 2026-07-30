#!/usr/bin/env python3
"""
Build Manuscript/Submission/RESS/main.tex from manuscript_full_text.md.

Output: Elsevier elsarticle-class LaTeX file for RESS submission.
Run from any directory; paths are relative to this script's location.

Usage:
    python build_ress_latex.py
"""

import re
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent          # Manuscript/
MD_PATH     = SCRIPT_DIR / "manuscript_full_text.md"
OUT_DIR     = REPO_ROOT / "Submission" / "RESS"
TEX_PATH    = OUT_DIR / "main.tex"
FIGURES_DIR = REPO_ROOT / "Figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Citation conversion  [1] or [5, 6] → \cite{ref01} ────────────────────────
_cite_re = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')

def convert_citations(text):
    r"""Replace numeric citation markers with \cite{} commands."""
    def repl(m):
        nums = [int(x.strip()) for x in m.group(1).split(',')]
        keys = ','.join(f'ref{n:02d}' for n in nums)
        return rf'\cite{{{keys}}}'
    return _cite_re.sub(repl, text)

# ── Inline formatting ─────────────────────────────────────────────────────────
_inline_re = re.compile(
    r'(\*\*[^*]+\*\*'       # **bold**
    r'|\*[^*]+\*'            # *italic*
    r'|`[^`]+`'              # `code`
    r'|\$\$[^$]+\$\$'        # $$inline-math$$
    r'|\$[^$\n]+\$'          # $inline-math$
    r')'
)

def latex_escape(text):
    """Escape LaTeX special chars (except math and cite already converted)."""
    # Don't escape inside existing \cite{} or math
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('^', r'\^{}')
    text = text.replace('~', r'\textasciitilde{}')
    return text

def inline_md_to_latex(text):
    """Convert inline Markdown formatting to LaTeX."""
    # Convert citations first (before escaping & or _)
    text = convert_citations(text)
    parts = _inline_re.split(text)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            inner = part[2:-2]
            out.append(rf'\textbf{{{latex_escape(inner)}}}')
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            inner = part[1:-1]
            out.append(rf'\textit{{{latex_escape(inner)}}}')
        elif part.startswith('`') and part.endswith('`') and len(part) > 2:
            inner = part[1:-1]
            out.append(rf'\texttt{{{latex_escape(inner)}}}')
        elif (part.startswith('$$') and part.endswith('$$')) or \
             (part.startswith('$') and part.endswith('$')):
            # Math: pass through as-is
            out.append(part)
        else:
            # Plain text — apply escaping but preserve \cite{} already done
            # Protect existing backslash sequences
            out.append(part.replace('&', r'\&').replace('%', r'\%'))
    return ''.join(out)

# ── Table conversion ──────────────────────────────────────────────────────────
def md_table_to_latex(lines, caption='', label=''):
    """Convert Markdown table lines to a LaTeX table environment."""
    rows = []
    col_aligns = None
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\|[-:\s|]+\|$', stripped):
            # Parse alignment from separator row
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            aligns = []
            for c in cells:
                if c.startswith(':') and c.endswith(':'):
                    aligns.append('c')
                elif c.endswith(':'):
                    aligns.append('r')
                else:
                    aligns.append('l')
            col_aligns = aligns
            continue
        cells = [c.strip() for c in stripped.strip('|').split('|')]
        rows.append(cells)

    if not rows:
        return ''

    n_cols = max(len(r) for r in rows)
    if col_aligns is None:
        col_aligns = ['l'] * n_cols

    col_spec = ''.join(col_aligns[:n_cols])

    lines_out = []
    lines_out.append(r'\begin{table}[!t]')
    lines_out.append(r'\centering')
    lines_out.append(r'\small')
    if caption:
        lines_out.append(rf'\caption{{{latex_escape(caption)}}}')
    if label:
        lines_out.append(rf'\label{{{label}}}')
    lines_out.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines_out.append(r'\toprule')

    for ri, row in enumerate(rows):
        cells_latex = []
        for ci in range(n_cols):
            cell = row[ci] if ci < len(row) else ''
            if ri == 0:
                cells_latex.append(rf'\textbf{{{inline_md_to_latex(cell)}}}')
            else:
                cells_latex.append(inline_md_to_latex(cell))
        lines_out.append(' & '.join(cells_latex) + r' \\')
        if ri == 0:
            lines_out.append(r'\midrule')

    lines_out.append(r'\bottomrule')
    lines_out.append(r'\end{tabular}')
    lines_out.append(r'\end{table}')
    return '\n'.join(lines_out)

# ── Section heading conversion ────────────────────────────────────────────────
_SECTION_MAP = {
    'abstract':             None,         # handled in frontmatter
    'references':           None,         # handled separately
    'i. introduction':      ('section',    'Introduction'),
    'ii.':                  ('section',    None),          # not in this MS
    'iii. methodology':     ('section',    'Methodology'),
    'iv. results':          ('section',    'Results'),
    'v. discussion':        ('section',    'Discussion and Implications'),
    'vi. conclusion':       ('section',    'Conclusion'),
}

def h2_to_section(title):
    key = title.strip().lower()
    for prefix, val in _SECTION_MAP.items():
        if key.startswith(prefix):
            return val
    return ('section', title)

def h3_to_subsection(title):
    # Remove leading letter prefix like "A. " or "C.1 "
    clean = re.sub(r'^[A-Z]\.\d*\s+', '', title).strip()
    clean = re.sub(r'^[A-Z]\.\s+', '', clean).strip()
    return ('subsection', clean)

def h4_to_subsubsection(title):
    return ('subsubsection', title)

# ── Main conversion ───────────────────────────────────────────────────────────
def convert():
    src = MD_PATH.read_text(encoding='utf-8')
    lines = src.split('\n')

    # ── Extract title (first H1) ──────────────────────────────────────────────
    title_line = ''
    for ln in lines:
        m = re.match(r'^# (.+)', ln.strip())
        if m:
            title_line = m.group(1).strip()
            break

    # ── Extract abstract ──────────────────────────────────────────────────────
    abstract_lines = []
    in_abstract = False
    abstract_done = False
    for ln in lines:
        s = ln.strip()
        if re.match(r'^## Abstract', s, re.I):
            in_abstract = True
            continue
        if in_abstract and not abstract_done:
            if s.startswith('## '):
                abstract_done = True
                in_abstract = False
                continue
            if s.startswith('>') or s.startswith('---') or not s:
                continue
            abstract_lines.append(s)

    abstract_text = '\n\n'.join(abstract_lines)

    # ── LaTeX preamble ────────────────────────────────────────────────────────
    tex = []
    tex.append(r'\documentclass[review,12pt]{elsarticle}')
    tex.append(r'% Elsevier elsarticle class — review mode (single column, line numbers)')
    tex.append(r'% Change to [preprint,12pt] for a clean preprint or [final,12pt] for final')
    tex.append('')
    tex.append(r'\usepackage{amsmath,amssymb}')
    tex.append(r'\usepackage{graphicx}')
    tex.append(r'\usepackage{booktabs}')
    tex.append(r'\usepackage{multirow}')
    tex.append(r'\usepackage{array}')
    tex.append(r'\usepackage{url}')
    tex.append(r'\usepackage{hyperref}')
    tex.append(r'\usepackage{xcolor}')
    tex.append(r'\usepackage{lineno}')
    tex.append(r'\modulolinenumbers[5]')
    tex.append('')
    tex.append(r'\journal{Reliability Engineering \& System Safety}')
    tex.append('')
    tex.append(r'\bibliographystyle{elsarticle-num}')
    tex.append('')
    tex.append(r'\begin{document}')
    tex.append(r'\begin{frontmatter}')
    tex.append('')

    # Title
    tex.append(rf'\title{{{title_line}}}')
    tex.append('')

    # Authors (single-blind — fill in before submission)
    tex.append(r'%% Author block — fill in before submission')
    tex.append(r'\author[inst1]{[Author Name]\corref{cor1}}')
    tex.append(r'\ead{spiceyoon@gmail.com}  % Replace with ETRI institutional email')
    tex.append(r'\cortext[cor1]{Corresponding author}')
    tex.append(r'\affiliation[inst1]{organization={Electronics and Telecommunications Research Institute (ETRI)},')
    tex.append(r'             city={Daejeon},')
    tex.append(r'             country={South Korea}}')
    tex.append('')

    # Abstract
    tex.append(r'\begin{abstract}')
    tex.append(inline_md_to_latex(abstract_text))
    tex.append(r'\end{abstract}')
    tex.append('')

    # Keywords
    tex.append(r'\begin{keyword}')
    tex.append(r'Remaining useful life prediction \sep')
    tex.append(r'Prognostics and health management \sep')
    tex.append(r'Turbofan engine \sep')
    tex.append(r'NASA CMAPSS \sep')
    tex.append(r'LSTM \sep')
    tex.append(r'Fault-mode gating \sep')
    tex.append(r'Normalization ablation')
    tex.append(r'\end{keyword}')
    tex.append('')
    tex.append(r'\end{frontmatter}')
    tex.append('')
    tex.append(r'\linenumbers')
    tex.append('')

    # ── Body content ──────────────────────────────────────────────────────────
    in_abstract   = False
    in_references = False
    skip_meta     = False
    i = 0
    first_h1_done = False
    pending_caption = ''
    pending_label   = ''
    table_counter   = 0

    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()

        # blank
        if not stripped:
            i += 1
            continue

        # blockquote (draft metadata) → skip
        if stripped.startswith('>'):
            i += 1
            continue

        # horizontal rule → skip
        if re.match(r'^-{3,}$', stripped):
            i += 1
            continue

        # End-of-file note
        if stripped.startswith('*End of manuscript'):
            break

        # ── H1 ───────────────────────────────────────────────────────────────
        if re.match(r'^# ', stripped):
            if not first_h1_done:
                first_h1_done = True  # title already in frontmatter
            i += 1
            continue

        # ── H2 ───────────────────────────────────────────────────────────────
        if re.match(r'^## ', stripped):
            h2_text = re.sub(r'^## ', '', stripped).strip()
            key_low = h2_text.lower()

            if key_low.startswith('abstract'):
                in_abstract = True
                i += 1
                continue
            else:
                in_abstract = False

            if key_low.startswith('references'):
                in_references = True
                tex.append('')
                tex.append(r'% References — generated from manuscript reference list')
                tex.append(r'% Replace \begin{thebibliography} with \bibliography{references}')
                tex.append(r'% if you use BibTeX (references.bib provided separately).')
                tex.append(r'\begin{thebibliography}{99}')
                i += 1
                continue

            result = h2_to_section(h2_text)
            if result is None:
                i += 1
                continue
            cmd, sec_title = result
            if sec_title is None:
                sec_title = h2_text
            tex.append('')
            tex.append(rf'\{cmd}{{{sec_title}}}')
            i += 1
            continue

        # ── H3 ───────────────────────────────────────────────────────────────
        if re.match(r'^### ', stripped):
            if in_abstract:
                i += 1
                continue
            h3_text = re.sub(r'^### ', '', stripped).strip()
            _, clean = h3_to_subsection(h3_text)
            tex.append('')
            tex.append(rf'\subsection{{{inline_md_to_latex(clean)}}}')
            i += 1
            continue

        # ── H4 ───────────────────────────────────────────────────────────────
        if re.match(r'^#### ', stripped):
            if in_abstract:
                i += 1
                continue
            h4_text = re.sub(r'^#### ', '', stripped).strip()
            tex.append('')
            tex.append(rf'\subsubsection{{{inline_md_to_latex(h4_text)}}}')
            i += 1
            continue

        # skip abstract body (already in frontmatter)
        if in_abstract:
            i += 1
            continue

        # ── Fenced code block ─────────────────────────────────────────────────
        if stripped.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            tex.append(r'\begin{verbatim}')
            tex.extend(code_lines)
            tex.append(r'\end{verbatim}')
            continue

        # ── Display math  $$ ... $$ ───────────────────────────────────────────
        if stripped == '$$':
            math_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                math_lines.append(lines[i].strip())
                i += 1
            i += 1
            tex.append(r'\begin{equation}')
            tex.extend(math_lines)
            tex.append(r'\end{equation}')
            continue
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            math_inner = stripped[2:-2].strip()
            tex.append(r'\begin{equation}')
            tex.append(math_inner)
            tex.append(r'\end{equation}')
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────────────
        if stripped.startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i].strip())
                i += 1
            table_counter += 1
            tbl_tex = md_table_to_latex(
                tbl_lines,
                caption=pending_caption,
                label=pending_label if pending_label else f'tab:{table_counter}'
            )
            tex.append('')
            tex.append(tbl_tex)
            tex.append('')
            pending_caption = ''
            pending_label   = ''
            continue

        # ── Table caption lines  **Table N. ...** ────────────────────────────
        m_tcap = re.match(r'^\*\*Table\s+([IVX]+)\.\s*(.*?)\*\*', stripped)
        if m_tcap:
            pending_caption = f'Table {m_tcap.group(1)}. {m_tcap.group(2).strip(".")}'
            pending_label   = f'tab:{m_tcap.group(1).lower()}'
            # Also emit as italic note before the table
            tex.append(rf'% {pending_caption}')
            i += 1
            continue

        # ── Figure placeholder  [FIGURE N: Title] ────────────────────────────
        m_fig = re.match(r'^\[FIGURE (\d+): (.+)\]$', stripped)
        if m_fig:
            fig_num   = m_fig.group(1)
            fig_title = m_fig.group(2)
            # Look ahead for **Fig. N.** caption on next non-blank line
            cap_line = ''
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r'^\*\*Fig\.', lines[j].strip()):
                cap_line = inline_md_to_latex(re.sub(r'\*\*', '', lines[j].strip()))
                i = j + 1
            else:
                cap_line = inline_md_to_latex(fig_title)
                i += 1
            fig_file = f'Fig{fig_num}'
            tex.append('')
            tex.append(r'\begin{figure}[!t]')
            tex.append(r'\centering')
            tex.append(rf'\includegraphics[width=\columnwidth]{{{fig_file}}}')
            tex.append(rf'\caption{{{cap_line}}}')
            tex.append(rf'\label{{fig:{fig_num}}}')
            tex.append(r'\end{figure}')
            tex.append('')
            continue

        # ── Figure caption line (if not already consumed above) ───────────────
        if re.match(r'^\*\*Fig\. \d', stripped):
            # Stand-alone caption line — emit as comment
            tex.append(f'% Caption: {stripped}')
            i += 1
            continue

        # ── Italic note line (table caption footnote starting with *) ─────────
        if stripped.startswith('*') and stripped.endswith('*') and '\n' not in stripped:
            tex.append(rf'% {stripped.strip("*")}')
            i += 1
            continue

        # ── Bullet list ────────────────────────────────────────────────────────
        if re.match(r'^[-*]\s+', stripped):
            # Collect consecutive bullet items
            items = []
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
                items.append(lines[i].strip()[2:])
                i += 1
            tex.append(r'\begin{itemize}')
            for item in items:
                tex.append(rf'  \item {inline_md_to_latex(item)}')
            tex.append(r'\end{itemize}')
            continue

        # ── Numbered list ──────────────────────────────────────────────────────
        nm = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if nm and not re.match(r'^\[\d+\]', stripped):  # not a reference
            items = []
            while i < len(lines):
                nm2 = re.match(r'^\d+\.\s+(.*)', lines[i].strip())
                if nm2:
                    items.append(nm2.group(1))
                    i += 1
                else:
                    break
            tex.append(r'\begin{enumerate}')
            for item in items:
                tex.append(rf'  \item {inline_md_to_latex(item)}')
            tex.append(r'\end{enumerate}')
            continue

        # ── Reference entries  [1] Author... ─────────────────────────────────
        if in_references and re.match(r'^\[\d+\]', stripped):
            m_ref = re.match(r'^\[(\d+)\]\s+(.*)', stripped)
            if m_ref:
                ref_num = int(m_ref.group(1))
                ref_body = m_ref.group(2)
                # Convert *italic* journal names to LaTeX
                ref_body = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', ref_body)
                tex.append(rf'\bibitem{{ref{ref_num:02d}}}')
                tex.append(f'  {ref_body}')
            i += 1
            continue

        # ── Author/affiliation meta lines in header → skip ────────────────────
        if re.match(r'^\*\*(Authors?|Affiliation|Correspondence):', stripped):
            i += 1
            continue

        # ── Compiled/Status lines from MS header → skip ───────────────────────
        if re.match(r'^\*\*(Compiled|Status|Source|Section):', stripped):
            i += 1
            continue

        # ── Normal paragraph ───────────────────────────────────────────────────
        tex.append('')
        tex.append(inline_md_to_latex(stripped))
        i += 1

    # Close thebibliography if we opened it
    if in_references:
        tex.append(r'\end{thebibliography}')

    tex.append('')
    tex.append(r'\end{document}')

    # ── Write output ──────────────────────────────────────────────────────────
    output = '\n'.join(tex)
    TEX_PATH.write_text(output, encoding='utf-8')
    print(f"Done → {TEX_PATH}")
    print(f"       {len(tex)} lines of LaTeX generated")
    print()
    print("Next steps:")
    print("  1. Copy Figures/*.png → Submission/RESS/  (or use \\graphicspath)")
    print("  2. Download elsarticle.cls from CTAN or Elsevier and place in Submission/RESS/")
    print("  3. Compile: pdflatex main.tex  (twice for cross-refs)")
    print("  4. Review output and adjust table widths / figure sizes as needed")
    print("  5. Fill in \\author{} and \\ead{} with ETRI institutional email")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    convert()
