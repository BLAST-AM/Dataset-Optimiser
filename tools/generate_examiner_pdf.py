from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


# ---------------------------
# Drawing helpers (matplotlib)
# ---------------------------

def _new_canvas(title: str, *, w: float = 12, h: float = 7) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        title,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
        family="DejaVu Sans",
    )


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#f7fbff",
    ec: str = "#1b2a41",
    lw: float = 1.8,
    fs: int = 11,
) -> None:
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(r)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        family="DejaVu Sans",
        wrap=True,
    )


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, text: str = "") -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.5,
        color="#1b2a41",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1b2a41",
            family="DejaVu Sans",
        )


def draw_dfd_level0(out: Path) -> Path:
    _new_canvas("DFD Level 0 (Context Diagram)")
    ax = plt.gca()

    _box(ax, 0.06, 0.42, 0.22, 0.16, "User\\n(Browser)", fc="#fff7e6", ec="#8a5a00")
    _box(ax, 0.36, 0.40, 0.28, 0.20, "Dataset Optimiser\\nFlask Web App", fc="#e8f5e9", ec="#1b5e20", fs=12)
    _box(ax, 0.72, 0.62, 0.23, 0.14, "Data Store\\nuploads/\\n(CSV files)", fc="#f3e5f5", ec="#4a148c")
    _box(ax, 0.72, 0.28, 0.23, 0.14, "Data Store\\nstatic/images/\\n(plots)", fc="#e3f2fd", ec="#0d47a1")

    _arrow(ax, 0.28, 0.50, 0.36, 0.50, "Upload CSV")
    _arrow(ax, 0.36, 0.46, 0.28, 0.46, "Dashboard / Results")

    _arrow(ax, 0.64, 0.56, 0.72, 0.68, "Save CSV")
    _arrow(ax, 0.64, 0.44, 0.72, 0.35, "Save images")
    _arrow(ax, 0.72, 0.66, 0.64, 0.52, "Read CSV")

    ax.text(
        0.02,
        0.06,
        "System boundary: everything inside the Flask app.",
        fontsize=10,
        color="#333333",
        family="DejaVu Sans",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def draw_dfd_level1(out: Path) -> Path:
    _new_canvas("DFD Level 1 (Main Processes)", w=12, h=8)
    ax = plt.gca()

    # Left actor
    _box(ax, 0.05, 0.78, 0.20, 0.12, "User", fc="#fff7e6", ec="#8a5a00")

    # Processes
    proc_w, proc_h = 0.36, 0.10
    x = 0.30
    ys = [0.80, 0.66, 0.52, 0.38, 0.24, 0.10]
    labels = [
        "1. Upload CSV\\nPOST /upload",
        "2. Analyze & Report\\n(results.html)",
        "3. Clean Dataset\\nPOST /clean",
        "4. Optimize Dataset\\nPOST /optimize",
        "5. Evaluate Model\\nPOST /evaluate",
        "6. Download Output\\nGET /download/<file>",
    ]
    for y, lab in zip(ys, labels):
        _box(ax, x, y, proc_w, proc_h, lab, fc="#e8f5e9", ec="#1b5e20", fs=10.5)

    # Stores
    _box(ax, 0.72, 0.68, 0.25, 0.14, "uploads/\\noriginal + clean_ + opt_ CSV", fc="#f3e5f5", ec="#4a148c", fs=10.5)
    _box(ax, 0.72, 0.30, 0.25, 0.14, "static/images/\\nplots + confusion matrix", fc="#e3f2fd", ec="#0d47a1", fs=10.5)

    # Flows from user to processes
    _arrow(ax, 0.25, 0.84, 0.30, 0.84, "file")
    _arrow(ax, 0.25, 0.70, 0.30, 0.70, "view")
    _arrow(ax, 0.25, 0.56, 0.30, 0.56, "strategy")
    _arrow(ax, 0.25, 0.42, 0.30, 0.42, "options")
    _arrow(ax, 0.25, 0.28, 0.30, 0.28, "target + model")
    _arrow(ax, 0.25, 0.14, 0.30, 0.14, "download")

    # Flows to/from stores
    _arrow(ax, 0.66, 0.84, 0.72, 0.77, "save")
    _arrow(ax, 0.72, 0.73, 0.66, 0.70, "read")

    _arrow(ax, 0.66, 0.70, 0.72, 0.37, "write")

    _arrow(ax, 0.72, 0.75, 0.66, 0.56, "read")
    _arrow(ax, 0.66, 0.56, 0.72, 0.75, "write clean_")

    _arrow(ax, 0.72, 0.75, 0.66, 0.42, "read")
    _arrow(ax, 0.66, 0.42, 0.72, 0.75, "write opt_")

    _arrow(ax, 0.72, 0.75, 0.66, 0.28, "read")
    _arrow(ax, 0.66, 0.28, 0.72, 0.37, "write CM")

    _arrow(ax, 0.72, 0.75, 0.66, 0.14, "read")

    ax.text(
        0.02,
        0.04,
        "Tip: You can explain this as “Upload → Analyze → (Optional) Clean/Optimize/Evaluate → Download”.",
        fontsize=10,
        color="#333333",
        family="DejaVu Sans",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def draw_uml_usecase(out: Path) -> Path:
    _new_canvas("UML Use Case (What the user can do)", w=12, h=7)
    ax = plt.gca()

    # Actor stick figure (simple)
    ax.plot([0.12, 0.12], [0.60, 0.42], color="#1b2a41", lw=2)
    ax.add_patch(plt.Circle((0.12, 0.66), 0.03, fill=False, ec="#1b2a41", lw=2))
    ax.plot([0.08, 0.16], [0.55, 0.55], color="#1b2a41", lw=2)
    ax.plot([0.12, 0.09], [0.42, 0.35], color="#1b2a41", lw=2)
    ax.plot([0.12, 0.15], [0.42, 0.35], color="#1b2a41", lw=2)
    ax.text(0.12, 0.30, "User", ha="center", va="top", fontsize=11, family="DejaVu Sans")

    # System boundary
    ax.add_patch(Rectangle((0.28, 0.18), 0.66, 0.70, fill=False, ec="#1b2a41", lw=2))
    ax.text(0.61, 0.86, "Dataset Optimiser", ha="center", va="center", fontsize=12, fontweight="bold")

    # Use cases (ovals approximated with rounded boxes)
    def oval(cx: float, cy: float, text: str) -> None:
        _box(ax, cx - 0.15, cy - 0.05, 0.30, 0.10, text, fc="#f7fbff", ec="#1b2a41", fs=10)

    oval(0.48, 0.72, "Upload CSV")
    oval(0.72, 0.72, "View Dashboard")
    oval(0.48, 0.56, "Clean Dataset")
    oval(0.72, 0.56, "Optimize Dataset")
    oval(0.48, 0.40, "Evaluate Model")
    oval(0.72, 0.40, "Download CSV")

    # Associations
    for cy in [0.72, 0.56, 0.40]:
        _arrow(ax, 0.17, cy, 0.33, cy, "")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def draw_uml_sequence_overview(out: Path) -> Path:
    _new_canvas("UML Sequence (Upload → Dashboard)", w=12, h=7)
    ax = plt.gca()

    lifelines = [
        (0.10, "User"),
        (0.32, "Flask App"),
        (0.54, "uploads/"),
        (0.76, "static/images/"),
    ]

    top = 0.85
    bottom = 0.15

    for x, name in lifelines:
        _box(ax, x - 0.08, 0.88, 0.16, 0.06, name, fc="#ffffff", ec="#1b2a41", fs=10)
        ax.plot([x, x], [top, bottom], linestyle="--", color="#1b2a41", lw=1.2)

    steps = [
        ("POST /upload (CSV)", 0, 1),
        ("Save file", 1, 2),
        ("Read CSV", 1, 2),
        ("Write plots", 1, 3),
        ("Return results.html", 1, 0),
    ]

    y = 0.78
    for label, a, b in steps:
        x1, _ = lifelines[a]
        x2, _ = lifelines[b]
        _arrow(ax, x1, y, x2, y, label)
        y -= 0.12

    ax.text(
        0.02,
        0.06,
        "Explain in words: The user uploads CSV → the server saves it → reads it → generates plots → shows the dashboard.",
        fontsize=10,
        color="#333333",
        family="DejaVu Sans",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def draw_uml_activity_clean(out: Path) -> Path:
    _new_canvas("UML Activity (Cleaning)", w=12, h=7)
    ax = plt.gca()

    # Simple flowchart-like activity
    _box(ax, 0.42, 0.82, 0.16, 0.08, "Start", fc="#ffffff", ec="#1b2a41", fs=11)
    _box(ax, 0.30, 0.68, 0.40, 0.09, "Read CSV (uploads/<filename>)", fc="#f7fbff", ec="#1b2a41")

    _box(ax, 0.30, 0.54, 0.40, 0.10, "Choose strategy:\nDrop OR Mean/Mode OR KNN", fc="#fff7e6", ec="#8a5a00")

    _box(ax, 0.07, 0.36, 0.26, 0.12, "Drop rows\nwith missing", fc="#e8f5e9", ec="#1b5e20")
    _box(ax, 0.37, 0.36, 0.26, 0.12, "Impute mean (num)\nmode (text)", fc="#e8f5e9", ec="#1b5e20")
    _box(ax, 0.67, 0.36, 0.26, 0.12, "KNN impute (num)\nmode (text)", fc="#e8f5e9", ec="#1b5e20")

    _box(ax, 0.30, 0.18, 0.40, 0.10, "Save clean_<filename>.csv\nRender export.html", fc="#f3e5f5", ec="#4a148c")
    _box(ax, 0.42, 0.06, 0.16, 0.08, "End", fc="#ffffff", ec="#1b2a41", fs=11)

    _arrow(ax, 0.50, 0.82, 0.50, 0.77)
    _arrow(ax, 0.50, 0.68, 0.50, 0.64)
    _arrow(ax, 0.50, 0.54, 0.20, 0.48, "drop")
    _arrow(ax, 0.50, 0.54, 0.50, 0.48, "mean")
    _arrow(ax, 0.50, 0.54, 0.80, 0.48, "knn")

    _arrow(ax, 0.20, 0.36, 0.50, 0.28)
    _arrow(ax, 0.50, 0.36, 0.50, 0.28)
    _arrow(ax, 0.80, 0.36, 0.50, 0.28)

    _arrow(ax, 0.50, 0.18, 0.50, 0.14)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def draw_uml_component(out: Path) -> Path:
    _new_canvas("UML Component (High-Level Architecture)", w=12, h=7)
    ax = plt.gca()

    _box(ax, 0.06, 0.70, 0.22, 0.14, "Browser", fc="#fff7e6", ec="#8a5a00")
    _box(ax, 0.36, 0.66, 0.28, 0.18, "Flask App\n(app.py)", fc="#e8f5e9", ec="#1b5e20", fs=12)

    _box(ax, 0.70, 0.72, 0.25, 0.12, "uploads/\nCSV files", fc="#f3e5f5", ec="#4a148c")
    _box(ax, 0.70, 0.54, 0.25, 0.12, "static/images/\nPNG plots", fc="#e3f2fd", ec="#0d47a1")

    _box(ax, 0.36, 0.42, 0.28, 0.12, "pandas / numpy\nData processing", fc="#f7fbff", ec="#1b2a41")
    _box(ax, 0.36, 0.26, 0.28, 0.12, "scikit-learn\nImpute & Evaluate", fc="#f7fbff", ec="#1b2a41")
    _box(ax, 0.36, 0.10, 0.28, 0.12, "matplotlib / seaborn\nPlots", fc="#f7fbff", ec="#1b2a41")

    _arrow(ax, 0.28, 0.77, 0.36, 0.77, "HTTP")
    _arrow(ax, 0.64, 0.76, 0.70, 0.78, "read/write")
    _arrow(ax, 0.64, 0.72, 0.70, 0.60, "write")

    _arrow(ax, 0.50, 0.66, 0.50, 0.54, "calls")
    _arrow(ax, 0.50, 0.66, 0.50, 0.38, "calls")
    _arrow(ax, 0.50, 0.66, 0.50, 0.22, "calls")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------
# PDF generation (ReportLab)
# ---------------------------

def build_examiner_pdf(output_pdf: Path, assets_dir: Path) -> None:
    styles = getSampleStyleSheet()

    title = styles["Title"]
    h = styles["Heading2"]
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Dataset Optimiser Diagrams",
    )

    def add_diagram(story, heading: str, explain: str, img_path: Path) -> None:
        story.append(Paragraph(heading, h))
        story.append(Paragraph(explain, body))

        # Fit image nicely on page
        max_w = A4[0] - doc.leftMargin - doc.rightMargin
        # Heuristic height cap for readability
        max_h = A4[1] - doc.topMargin - doc.bottomMargin - 7.0 * cm

        story.append(Image(str(img_path), width=max_w, height=max_h, kind="proportional"))
        story.append(Spacer(1, 10))

    story = []
    story.append(Paragraph("Dataset Optimiser", title))
    story.append(
        Paragraph(
            "Clear & concise DFD and UML diagrams (examiner-friendly).", body
        )
    )
    story.append(
        Paragraph(
            "Project summary: A user uploads a CSV, the app analyzes it, can clean/optimize/evaluate, then lets the user download results.",
            body,
        )
    )
    story.append(Spacer(1, 10))

    add_diagram(
        story,
        "DFD Level 0 — Context",
        "Shows the whole system as a single process with external user and file stores.",
        assets_dir / "dfd_level0.png",
    )
    story.append(PageBreak())

    add_diagram(
        story,
        "DFD Level 1 — Main Processes",
        "Breaks the system into the main features: upload, analyze, clean, optimize, evaluate, download.",
        assets_dir / "dfd_level1.png",
    )
    story.append(PageBreak())

    add_diagram(
        story,
        "UML Use Case",
        "Explains what the user can do in the system (features from the user’s point of view).",
        assets_dir / "uml_usecase.png",
    )
    story.append(PageBreak())

    add_diagram(
        story,
        "UML Sequence — Upload to Dashboard",
        "Shows the step-by-step interaction when uploading a dataset and viewing the dashboard.",
        assets_dir / "uml_sequence_overview.png",
    )
    story.append(PageBreak())

    add_diagram(
        story,
        "UML Activity — Cleaning",
        "Shows the cleaning logic: choose a strategy → apply it → save cleaned CSV → show download.",
        assets_dir / "uml_activity_clean.png",
    )
    story.append(PageBreak())

    add_diagram(
        story,
        "UML Component — Architecture",
        "Shows the main parts used by the app: Flask routes + data processing + ML + plotting + file stores.",
        assets_dir / "uml_component.png",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs" / "diagrams_assets"

    # Generate diagram images
    draw_dfd_level0(assets / "dfd_level0.png")
    draw_dfd_level1(assets / "dfd_level1.png")
    draw_uml_usecase(assets / "uml_usecase.png")
    draw_uml_sequence_overview(assets / "uml_sequence_overview.png")
    draw_uml_activity_clean(assets / "uml_activity_clean.png")
    draw_uml_component(assets / "uml_component.png")

    # Build PDF
    out_pdf = root / "docs" / "diagrams_examiner.pdf"
    build_examiner_pdf(out_pdf, assets)
    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
