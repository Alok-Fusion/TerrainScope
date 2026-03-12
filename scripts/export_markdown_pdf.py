from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1b2630"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=25,
            textColor=colors.HexColor("#13202a"),
            spaceBefore=4,
            spaceAfter=10,
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#13202a"),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading3Custom",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#13202a"),
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeCustom",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=9.6,
            backColor=colors.HexColor("#f5f1ea"),
            borderColor=colors.HexColor("#d8cfc3"),
            borderWidth=0.5,
            borderPadding=8,
            borderRadius=4,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=9.8,
            textColor=colors.HexColor("#586776"),
            alignment=1,
            spaceBefore=2,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ListCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            leftIndent=12,
            spaceAfter=2,
            textColor=colors.HexColor("#1b2630"),
        )
    )
    return styles


def inline_to_rl_markup(node) -> str:
    if isinstance(node, NavigableString):
        return html.escape(str(node))

    if not isinstance(node, Tag):
        return ""

    tag = node.name.lower()
    inner = "".join(inline_to_rl_markup(child) for child in node.children)

    if tag in {"strong", "b"}:
        return f"<b>{inner}</b>"
    if tag in {"em", "i"}:
        return f"<i>{inner}</i>"
    if tag == "code":
        return f'<font name="Courier">{inner}</font>'
    if tag == "a":
        href = html.escape(node.get("href", ""))
        return f'<a href="{href}">{inner}</a>'
    if tag == "br":
        return "<br/>"
    if tag == "img":
        return ""
    return inner


def paragraph_from_tag(tag: Tag, style) -> Paragraph:
    return Paragraph("".join(inline_to_rl_markup(child) for child in tag.children), style)


def image_flowables(img_tag: Tag, base_dir: Path, max_width: float, max_height: float, styles) -> list:
    src = img_tag.get("src", "")
    if not src:
        return []

    image_path = (base_dir / src).resolve()
    if not image_path.exists():
        return [Paragraph(f"Missing image: {html.escape(src)}", styles["Body"])]

    image_reader = ImageReader(str(image_path))
    width, height = image_reader.getSize()
    scale = min(max_width / width, max_height / height, 1.0)
    image = Image(str(image_path), width=width * scale, height=height * scale)
    image.hAlign = "CENTER"

    elements = [image]
    alt = img_tag.get("alt", "").strip()
    if alt:
        elements.append(Paragraph(alt, styles["Caption"]))
    else:
        elements.append(Spacer(1, 8))
    return elements


def table_flowable(table_tag: Tag, styles, available_width: float):
    rows = []
    for tr in table_tag.find_all("tr"):
        row = []
        for cell in tr.find_all(["th", "td"]):
            row.append(Paragraph("".join(inline_to_rl_markup(child) for child in cell.children), styles["Body"]))
        if row:
            rows.append(row)

    if not rows:
        return None

    col_count = max(len(row) for row in rows)
    col_widths = [available_width / col_count] * col_count
    table = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efe6da")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#13202a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.2),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9cfc2")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def list_flowable(list_tag: Tag, styles, ordered: bool):
    items = []
    for index, item in enumerate(list_tag.find_all("li", recursive=False), start=1):
        text = "".join(inline_to_rl_markup(child) for child in item.children)
        paragraph = Paragraph(text, styles["ListCustom"])
        if ordered:
            items.append(ListItem(paragraph, value=index))
        else:
            items.append(ListItem(paragraph))

    if not items:
        return None

    return ListFlowable(
        items,
        bulletType="1" if ordered else "bullet",
        start="1",
        leftIndent=8,
        bulletFontName="Helvetica",
        bulletFontSize=10,
        bulletOffsetY=2,
    )


def html_to_flowables(soup: BeautifulSoup, styles, base_dir: Path, available_width: float) -> list:
    elements = []
    max_image_width = available_width
    max_image_height = 2.75 * inch

    for node in soup.div.children:
        if isinstance(node, NavigableString):
            if str(node).strip():
                elements.append(Paragraph(html.escape(str(node).strip()), styles["Body"]))
            continue

        if not isinstance(node, Tag):
            continue

        name = node.name.lower()

        if name == "h1":
            elements.append(Paragraph(node.get_text(strip=True), styles["Heading1Custom"]))
            continue
        if name == "h2":
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(node.get_text(strip=True), styles["Heading2Custom"]))
            continue
        if name == "h3":
            elements.append(Paragraph(node.get_text(strip=True), styles["Heading3Custom"]))
            continue
        if name == "p":
            images = node.find_all("img", recursive=False)
            non_image_text = "".join(child.get_text(strip=True) for child in node.children if isinstance(child, Tag) and child.name != "img")
            if images and not non_image_text and not any(isinstance(child, NavigableString) and child.strip() for child in node.children):
                for image_tag in images:
                    elements.extend(image_flowables(image_tag, base_dir, max_image_width, max_image_height, styles))
            else:
                text = "".join(inline_to_rl_markup(child) for child in node.children).strip()
                if text:
                    elements.append(Paragraph(text, styles["Body"]))
            continue
        if name == "ul":
            flowable = list_flowable(node, styles, ordered=False)
            if flowable:
                elements.append(flowable)
                elements.append(Spacer(1, 6))
            continue
        if name == "ol":
            flowable = list_flowable(node, styles, ordered=True)
            if flowable:
                elements.append(flowable)
                elements.append(Spacer(1, 6))
            continue
        if name == "pre":
            code_text = node.get_text("\n", strip=False).rstrip()
            if code_text:
                elements.append(Preformatted(code_text, styles["CodeCustom"]))
            continue
        if name == "table":
            table = table_flowable(node, styles, available_width)
            if table:
                elements.append(table)
                elements.append(Spacer(1, 10))
            continue
        if name == "hr":
            elements.append(Spacer(1, 4))
            elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cfbfae")))
            elements.append(Spacer(1, 8))
            continue
        if name == "div":
            elements.extend(html_to_flowables(BeautifulSoup(f"<div>{node.decode_contents()}</div>", "html.parser"), styles, base_dir, available_width))
            continue

    return elements


def markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    source_text = input_path.read_text(encoding="utf-8")
    source_text = re.sub(r"\n\\newpage\s*\n", "\n<hr />\n", source_text)
    html_fragment = markdown.markdown(source_text, extensions=["tables", "fenced_code"])
    soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=input_path.stem,
        author="OpenAI Codex",
    )
    styles = build_styles()
    available_width = A4[0] - doc.leftMargin - doc.rightMargin
    elements = html_to_flowables(soup, styles, input_path.parent, available_width)
    doc.build(elements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a markdown file to PDF using ReportLab.")
    parser.add_argument("input_path", type=str, help="Path to the markdown file.")
    parser.add_argument("output_path", type=str, nargs="?", default=None, help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else input_path.with_suffix(".pdf")
    markdown_to_pdf(input_path, output_path)
    print(f"PDF written to {output_path}")


if __name__ == "__main__":
    main()
