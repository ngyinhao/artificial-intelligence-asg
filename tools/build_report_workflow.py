"""Build the report workflow diagram with reader-facing terminology."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("output/ComplaintCompass_Report_media/media/image8.png")
W, H = 1600, 1120
BG = "white"
INK = "#1f2933"
BLUE = "#0f4f6c"
PALE_BLUE = "#eaf3f7"
PALE_GOLD = "#fff5d6"
LINE = "#53636d"


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


TITLE = font(34, True)
LABEL = font(26, True)
BODY = font(25)
SMALL = font(22)


def wrapped(draw, text, box, fnt, fill=INK, spacing=5):
    x1, y1, x2, y2 = box
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= x2 - x1 - 28:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    heights = [draw.textbbox((0, 0), value, font=fnt)[3] for value in lines]
    total = sum(heights) + spacing * max(0, len(lines) - 1)
    y = y1 + max(12, (y2 - y1 - total) / 2)
    for value, height in zip(lines, heights):
        width = draw.textbbox((0, 0), value, font=fnt)[2]
        draw.text((x1 + (x2 - x1 - width) / 2, y), value, font=fnt, fill=fill)
        y += height + spacing


def box(draw, coords, text, fill="white", fnt=BODY):
    draw.rounded_rectangle(coords, radius=12, fill=fill, outline=LINE, width=3)
    wrapped(draw, text, coords, fnt)


def arrow(draw, start, end):
    draw.line([start, end], fill=LINE, width=5)
    ex, ey = end
    draw.polygon([(ex, ey), (ex - 16, ey - 10), (ex - 16, ey + 10)], fill=LINE)


def section(draw, top, bottom, title):
    draw.rounded_rectangle((55, top, W - 55, bottom), radius=18, fill=PALE_BLUE, outline=LINE, width=3)
    draw.text((82, top + 20), title, font=TITLE, fill=BLUE)


image = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(image)

section(draw, 35, 345, "A. DATA PREPARATION")
boxes = [(90, 120, 370, 255), (455, 120, 735, 255), (820, 120, 1100, 255), (1185, 120, 1510, 255)]
texts = [
    "Official CFPB complaint narratives",
    "Normalize text and remove exact duplicates",
    "Balance six classes and connect narratives with cosine similarity >= 0.90",
    "Keep each group in one 70/15/15 split; verify zero threshold crossings",
]
for coords, text in zip(boxes, texts):
    box(draw, coords, text, PALE_GOLD if "near" in text else "white")
for left, right in zip(boxes, boxes[1:]):
    arrow(draw, (left[2], (left[1] + left[3]) // 2), (right[0], (right[1] + right[3]) // 2))
draw.text((95, 285), "18,000 records: 12,600 train | 2,700 validation | 2,700 test", font=LABEL, fill=INK)

section(draw, 375, 735, "B. MODEL DEVELOPMENT AND EVALUATION")
top_boxes = [(90, 465, 440, 590), (505, 465, 855, 590), (920, 465, 1510, 590)]
top_texts = [
    "Train and tune three base models using training/CV",
    "Compare base models on validation",
    "Select weighted-ensemble weights, ARUF settings, and the application default on validation",
]
for coords, text in zip(top_boxes, top_texts):
    box(draw, coords, text, PALE_GOLD if "Select" in text else "white", SMALL)
for left, right in zip(top_boxes, top_boxes[1:]):
    arrow(draw, (left[2], (left[1] + left[3]) // 2), (right[0], (right[1] + right[3]) // 2))
box(draw, (250, 630, 760, 700), "Refit base models on training + validation", "white", SMALL)
box(draw, (840, 630, 1350, 700), "Current test comparison; combination results are post-hoc", PALE_GOLD, SMALL)
arrow(draw, (760, 665), (840, 665))

section(draw, 765, 1080, "C. APPLICATION INFERENCE")
app_boxes = [(90, 855, 390, 990), (465, 855, 765, 990), (840, 855, 1140, 990), (1215, 855, 1510, 990)]
app_texts = [
    "User enters 20-2,000 valid characters",
    "Normalize input in memory",
    "Run the selected saved model",
    "Show category, model confidence, and top three",
]
for coords, text in zip(app_boxes, app_texts):
    box(draw, coords, text, "white", SMALL)
for left, right in zip(app_boxes, app_boxes[1:]):
    arrow(draw, (left[2], (left[1] + left[3]) // 2), (right[0], (right[1] + right[3]) // 2))
draw.text((95, 1020), "Submitted narratives are processed locally and are not intentionally stored.", font=LABEL, fill=INK)

OUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUT, dpi=(180, 180))
