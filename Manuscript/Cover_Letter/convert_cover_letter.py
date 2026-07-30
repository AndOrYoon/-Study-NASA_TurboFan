#!/usr/bin/env python3
"""Convert Cover_Letter_draft.md to Cover_letter(IEE_TIE).docx"""

import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

MD_PATH   = Path(__file__).parent / "Cover_Letter_draft.md"
DOCX_PATH = Path(__file__).parent / "Cover_letter(IEEE_TII).docx"

BOLD_PATTERN = re.compile(r'\*\*([^*]+)\*\*')

def add_inline(para, text):
    parts = BOLD_PATTERN.split(text)
    for i, part in enumerate(parts):
        if not part:
            continue
        run = para.add_run(part)
        if i % 2 == 1:   # odd index = captured bold group
            run.bold = True

def set_font(run, size=12):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)

def para_font(para, size=12):
    for run in para.runs:
        set_font(run, size)

def convert():
    doc = Document()

    # Page margins — standard letter: 1" all sides
    for sec in doc.sections:
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin   = Inches(1.25)
        sec.right_margin  = Inches(1.25)

    # Normal style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)

    with open(MD_PATH, encoding='utf-8') as f:
        lines = f.read().split('\n')

    # States
    in_notes = False     # skip trailing notes block
    skip_next_hr = False # skip decorative --- separators between header and body

    i = 0
    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()

        # Skip blank lines — add spacing via paragraph_format instead
        if not stripped:
            i += 1
            continue

        # Detect start of trailing notes blockquote
        if stripped.startswith('> **Pre-submission checklist') or stripped.startswith('> **Notes for revision'):
            in_notes = True
        if in_notes:
            i += 1
            continue

        # Skip H1 title (document title only)
        if stripped.startswith('# '):
            i += 1
            continue

        # Skip blockquote metadata lines (status/action required)
        if stripped.startswith('>'):
            i += 1
            continue

        # Horizontal rules — use as paragraph spacers, not visible lines
        if re.match(r'^-{3,}$', stripped):
            doc.add_paragraph()
            i += 1
            continue

        # --- Sender block: name / affiliation / email (plain, right-aligned) ---
        if stripped == '[Author Name]':
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run('[Author Name]')
            set_font(r)
            i += 1
            continue

        # Date line
        if stripped.startswith('July '):
            p = doc.add_paragraph()
            r = p.add_run(stripped)
            set_font(r)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after  = Pt(12)
            i += 1
            continue

        # Salutation
        if stripped.startswith('Dear '):
            p = doc.add_paragraph()
            r = p.add_run(stripped)
            set_font(r)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(12)
            i += 1
            continue

        # Closing "Sincerely,"
        if stripped == 'Sincerely,':
            p = doc.add_paragraph()
            r = p.add_run('Sincerely,')
            set_font(r)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after  = Pt(36)  # space for signature
            i += 1
            continue

        # Body paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(8)
        p.paragraph_format.first_line_indent = Pt(0)
        add_inline(p, stripped)
        para_font(p)
        i += 1

    doc.save(DOCX_PATH)
    print(f"Done → {DOCX_PATH}")

if __name__ == '__main__':
    convert()
