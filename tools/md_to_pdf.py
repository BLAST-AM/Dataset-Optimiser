from __future__ import annotations

import html
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


@dataclass
class Block:
    kind: str  # heading|para|code|hr
    text: str
    level: int = 0


def parse_markdown(md: str) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []

    in_code = False
    code_lang = ""
    code_lines: list[str] = []

    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        if not para_lines:
            return
        text = " ".join(s.strip() for s in para_lines).strip()
        if text:
            blocks.append(Block(kind="para", text=text))
        para_lines = []

    for raw in lines:
        line = raw.rstrip("\n")

        m_fence = re.match(r"^```(.*)$", line.strip())
        if m_fence:
            if in_code:
                # close fence
                blocks.append(
                    Block(kind="code", text="\n".join(code_lines).rstrip("\n"), level=0)
                )
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                flush_para()
                in_code = True
                code_lang = m_fence.group(1).strip()
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        # Horizontal rule
        if line.strip() in {"---", "***"}:
            flush_para()
            blocks.append(Block(kind="hr", text=""))
            continue

        # Headings
        m_h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m_h:
            flush_para()
            level = len(m_h.group(1))
            text = m_h.group(2).strip()
            blocks.append(Block(kind="heading", text=text, level=level))
            continue

        # Blank line separates paragraphs
        if not line.strip():
            flush_para()
            continue

        # Bullets: keep readable
        if line.lstrip().startswith("- "):
            para_lines.append("• " + line.lstrip()[2:].strip())
            continue

        para_lines.append(line)

    flush_para()

    # If file ends while still in code, flush as code
    if in_code and code_lines:
        blocks.append(Block(kind="code", text="\n".join(code_lines).rstrip("\n")))

    return blocks


def build_pdf(input_md: Path, output_pdf: Path) -> None:
    md = input_md.read_text(encoding="utf-8")
    blocks = parse_markdown(md)

    styles = getSampleStyleSheet()

    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.3,
        leading=10.5,
        spaceBefore=6,
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=input_md.stem,
    )

    story = []

    story.append(Paragraph("Dataset Optimiser — DFD + UML Diagrams", h1))
    story.append(Paragraph("(Exported from diagrams.md)", body))
    story.append(Spacer(1, 10))

    for b in blocks:
        if b.kind == "heading":
            if b.level <= 1:
                story.append(Paragraph(html.escape(b.text), h1))
            elif b.level == 2:
                story.append(Paragraph(html.escape(b.text), h2))
            else:
                story.append(Paragraph(html.escape(b.text), h3))
            story.append(Spacer(1, 6))
        elif b.kind == "para":
            # Paragraph supports a tiny subset of HTML; escape everything.
            story.append(Paragraph(html.escape(b.text), body))
        elif b.kind == "code":
            # Keep as verbatim text for easy copy/paste.
            story.append(Preformatted(b.text, code_style, dedent=0))
        elif b.kind == "hr":
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph(html.escape(b.text), body))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python tools/md_to_pdf.py <input.md> <output.pdf>")
        return 2

    input_md = Path(argv[1]).resolve()
    output_pdf = Path(argv[2]).resolve()

    if not input_md.exists():
        print(f"Input not found: {input_md}")
        return 2

    build_pdf(input_md, output_pdf)
    print(str(output_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
