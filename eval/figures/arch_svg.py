"""Report Figure 1: system architecture as a hand-laid-out SVG (no crossing edges).
Usage (from the repository root): python eval/figures/arch_svg.py  → eval/figures/fig_architecture.svg"""
from pathlib import Path
W, H = 1300, 748
BW, BH = 240, 100
COL = [200, 480, 760, 1040]          # box x positions (row B left→right, row C right→left)
YA, YB, YC, YD = 72, 262, 445, 595   # row y (top of boxes)
INK, INK2, LINE = "#1f1f1e", "#5f5e5a", "#8a8984"
STY = {
    "plain":  ("#f6f6f4", "#c9c8c2"),
    "retr":   ("#e8f0fb", "#2a78d6"),
    "ask":    ("#fdeee6", "#eb6834"),
    "check":  ("#e6f5ee", "#1baf7a"),
    "user":   ("#ffffff", "#5f5e5a"),
    "rw":     ("#efedfb", "#6250d6"),
}
out = []
def box(x, y, title, lines, kind="plain", w=BW, h=BH, tag=None):
    fill, stroke = STY[kind]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = 1 + len(lines); top = y + h / 2 - (n - 1) * 11.5
    out.append(f'<text x="{x + w/2}" y="{top + 7}" text-anchor="middle" font-size="21" font-weight="600" fill="{INK}">{title}</text>')
    for i, l in enumerate(lines):
        out.append(f'<text x="{x + w/2}" y="{top + 7 + 23*(i+1)}" text-anchor="middle" font-size="18" fill="{INK2}">{l}</text>')
    if tag:
        tw = 12 + 10 * len(tag)
        out.append(f'<rect x="{x + w - tw - 6}" y="{y - 12}" width="{tw}" height="24" rx="12" fill="{stroke}"/>')
        out.append(f'<text x="{x + w - tw/2 - 6}" y="{y + 5.5}" text-anchor="middle" font-size="15" font-weight="600" fill="#ffffff">{tag}</text>')
def arrow(points, dashed=False, label=None, lx=None, ly=None, anchor="middle"):
    d = "M " + " L ".join(f"{px} {py}" for px, py in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2"{dash} marker-end="url(#ah)"/>')
    if label:
        out.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="17" font-style="italic" fill="{INK2}">{label}</text>')

# ---- offline knowledge base (row A), directly above the first three pipeline columns
out.append(f'<rect x="{COL[0]-20}" y="{YA-36}" width="{COL[2]+BW-COL[0]+40}" height="{BH+56}" rx="14" fill="none" stroke="#c9c8c2" stroke-width="1.3" stroke-dasharray="7 6"/>')
out.append(f'<text x="{COL[0]-4}" y="{YA-12}" font-size="18" font-weight="600" fill="{INK2}">OFFLINE KNOWLEDGE BASE (Q1)</text>')
box(COL[0], YA, "Amazon Reviews 2023", ["3 categories", "15,000 products"])
box(COL[1], YA, "Clean &amp; chunk", ["cleaning · de-dup", "48,107 chunks"])
box(COL[2], YA, "Indexes", ["BM25 · bge-m3 FAISS", "+ metadata"])
arrow([(COL[0]+BW, YA+BH/2), (COL[1]-2, YA+BH/2)])
arrow([(COL[1]+BW, YA+BH/2), (COL[2]-2, YA+BH/2)])

# ---- per-turn pipeline
out.append(f'<text x="{COL[0]-4}" y="{YB-24}" font-size="18" font-weight="600" fill="{INK2}">EVERY DIALOGUE TURN (Q2–Q5)</text>')
box(COL[0], YB, "State tracking", ["need slots +", "English query"], kind="rw", tag="I2")
box(COL[1], YB, "Filters", ["price · audience · age", "material · date"], kind="rw", tag="I2")
box(COL[2], YB, "Hybrid retrieval", ["BM25 + dense", "MaxP → RRF"], kind="retr")
box(COL[3], YB, "Re-ranking", ["cross-encoder", "bge-reranker-v2-m3"], kind="retr")
box(COL[3], YC, "Decision", ["ask / recommend /", "none (entropy)"], kind="ask", tag="I1")
box(COL[2], YC, "Evidence check", ["constraints + LLM", "match, relax &amp; retry"], kind="check", tag="I3")
box(COL[1], YC, "Generation", ["JSON answer with", "product &amp; evidence IDs"])
box(COL[0], YC, "Claim self-check", ["verify each claim,", "prices vs metadata"], kind="check", tag="I3")
box(COL[3], YD, "Clarifying question", ["most informative attribute"], kind="ask", h=76)
# user (left column, spans rows B and C)
UX, UW = 20, 120
box(UX, YB, "User", ["English", "or Chinese"], kind="user", w=UW, h=YC + BH - YB)

ay = lambda y: y + BH / 2
arrow([(UX+UW, ay(YB)), (COL[0]-2, ay(YB))], label="need", lx=(UX+UW+COL[0])/2, ly=ay(YB)-8)
for i in range(3):
    arrow([(COL[i]+BW, ay(YB)), (COL[i+1]-2, ay(YB))])
arrow([(COL[2]+BW/2, YA+BH), (COL[2]+BW/2, YB-2)], dashed=True, label="search", lx=COL[2]+BW/2+8, ly=(YA+BH+YB)/2+4, anchor="start")
arrow([(COL[3]+BW/2, YB+BH), (COL[3]+BW/2, YC-2)])
arrow([(COL[3], ay(YC)), (COL[2]+BW+2, ay(YC))], label="rec.", lx=(COL[2]+BW+COL[3])/2, ly=ay(YC)-8)
arrow([(COL[2], ay(YC)), (COL[1]+BW+2, ay(YC))])
arrow([(COL[1], ay(YC)), (COL[0]+BW+2, ay(YC))])
arrow([(COL[0], ay(YC)), (UX+UW+2, ay(YC))], label="top-3", lx=(UX+UW+COL[0])/2, ly=ay(YC)-8)
arrow([(COL[3]+BW/2, YC+BH), (COL[3]+BW/2, YD-2)], label="ask", lx=COL[3]+BW/2+8, ly=(YC+BH+YD)/2+4, anchor="start")
LY = YD + 38
arrow([(COL[3], LY), (UX+UW/2, LY), (UX+UW/2, YC+BH+2)], dashed=True, label="the user answers → next turn", lx=(UX+UW/2+COL[3])/2, ly=LY-8)

# legend
lx, ly = UX + 4, H - 26
for i, (kind, name) in enumerate([("rw", "query understanding (I2)"), ("retr", "retrieval (Q2/Q3)"), ("ask", "clarification (I1)"), ("check", "grounding checks (I3)")]):
    fill, stroke = STY[kind]; x = lx + i * 255
    out.append(f'<rect x="{x}" y="{ly-15}" width="20" height="20" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    out.append(f'<text x="{x+28}" y="{ly+1}" font-size="17" fill="{INK2}">{name}</text>')


svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, 'Helvetica Neue', Arial, sans-serif">
<title>System architecture of the conversational RAG product recommender</title>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
''' + "\n".join(out) + "\n</svg>\n"
(Path(__file__).resolve().parent / "fig_architecture.svg").write_text(svg, encoding="utf-8")
