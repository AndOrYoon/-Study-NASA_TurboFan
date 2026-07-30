#!/usr/bin/env python3
"""Convert manuscript_full_text.md to manuscript_full_text.docx using python-docx."""

import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MD_PATH   = Path(__file__).parent / "manuscript_full_text.md"
DOCX_PATH = Path(__file__).parent / "manuscript_full_text.docx"

# ---------------------------------------------------------------------------
# Inline formatting helper
# ---------------------------------------------------------------------------
INLINE_PATTERN = re.compile(
    r'(\*\*[^*]+\*\*'          # **bold**
    r'|\*[^*]+\*'               # *italic*
    r'|`[^`]+`'                 # `code`
    r'|\$\$[^$]+\$\$'           # $$inline-math$$
    r'|\$[^$\n]+\$'             # $inline-math$
    r')'
)

def add_inline(para, text):
    """Add text to paragraph with inline bold/italic/code/math handling."""
    # Strip common leading/trailing pipes and whitespace artefacts
    parts = INLINE_PATTERN.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            run = para.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            run = para.add_run(part[1:-1])
            run.italic = True
        elif part.startswith('`') and part.endswith('`') and len(part) > 2:
            run = para.add_run(part[1:-1])
            run.font.name = 'Courier New'
            run.font.size = Pt(10)
        elif (part.startswith('$$') and part.endswith('$$')) or \
             (part.startswith('$') and part.endswith('$')):
            inner = part.strip('$')
            run = para.add_run(f'[{inner}]')
            run.italic = True
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        else:
            para.add_run(part)


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------
def add_shading(cell, fill_hex='DEEAF1'):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  fill_hex)
    tcPr.append(shd)


def process_table(doc, table_lines):
    rows = []
    for line in table_lines:
        # Skip separator rows  |---|:---:|
        if re.match(r'^\|[-:\s|]+\|$', line):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)

    if not rows:
        return

    max_cols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=max_cols)
    tbl.style = 'Table Grid'

    for ri, row in enumerate(rows):
        for ci in range(max_cols):
            cell_text = row[ci] if ci < len(row) else ''
            cell = tbl.cell(ri, ci)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after  = Pt(2)
            if ri == 0:
                add_shading(cell, 'BDD7EE')   # header: light blue
                run = p.add_run(cell_text)
                run.bold = True
                run.font.size = Pt(10)
            else:
                add_inline(p, cell_text)
                for run in p.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Code-block helper
# ---------------------------------------------------------------------------
def add_code_block(doc, code_text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after  = Pt(4)
    para.paragraph_format.left_indent  = Cm(0.5)
    run = para.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(8.5)
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  'F2F2F2')
    pPr.append(shd)


# ---------------------------------------------------------------------------
# Math-block helper
# ---------------------------------------------------------------------------
def add_math_block(doc, math_text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after  = Pt(4)
    run = para.add_run(math_text)
    run.italic = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


# ---------------------------------------------------------------------------
# Horizontal-rule helper
# ---------------------------------------------------------------------------
def add_hrule(doc):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after  = Pt(4)
    pPr  = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot  = OxmlElement('w:bottom')
    bot.set(qn('w:val'),   'single')
    bot.set(qn('w:sz'),    '6')
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bot)
    pPr.append(pBdr)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert():
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    # Normal style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    from docx.oxml.ns import qn as _qn
    normal._element.rPr.rFonts.set(_qn('w:eastAsia'), 'Times New Roman')

    # Heading styles
    for lvl in range(1, 5):
        h = doc.styles[f'Heading {lvl}']
        h.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)  # dark blue
        h.font.name = 'Calibri'

    with open(MD_PATH, encoding='utf-8') as f:
        lines = f.read().split('\n')

    i = 0
    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()

        # --- blank line ---
        if not stripped:
            i += 1
            continue

        # --- draft-metadata blockquotes (skip) ---
        if stripped.startswith('>'):
            i += 1
            continue

        # --- horizontal rule ---
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', stripped):
            add_hrule(doc)
            i += 1
            continue

        # --- headers ---
        hm = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if hm:
            lvl   = min(len(hm.group(1)), 4)
            title = hm.group(2).strip()
            doc.add_heading(title, level=lvl)
            i += 1
            continue

        # --- fenced code block ---
        if stripped.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1   # closing ```
            add_code_block(doc, '\n'.join(code_lines))
            continue

        # --- display math block  $$ ... $$ ---
        if stripped == '$$':
            math_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                math_lines.append(lines[i].strip())
                i += 1
            i += 1   # closing $$
            add_math_block(doc, '  '.join(math_lines))
            continue
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            add_math_block(doc, stripped[2:-2].strip())
            i += 1
            continue

        # --- table ---
        if stripped.startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i].strip())
                i += 1
            process_table(doc, tbl_lines)
            continue

        # --- bullet list ---
        if re.match(r'^[-*]\s+', stripped):
            para = doc.add_paragraph(style='List Bullet')
            add_inline(para, stripped[2:])
            i += 1
            continue

        # --- numbered list ---
        nm = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if nm:
            para = doc.add_paragraph(style='List Number')
            add_inline(para, nm.group(2))
            i += 1
            continue

        # --- normal paragraph ---
        para = doc.add_paragraph()
        add_inline(para, stripped)
        i += 1

    doc.save(DOCX_PATH)
    print(f"Done → {DOCX_PATH}")


if __name__ == '__main__':
    convert()
