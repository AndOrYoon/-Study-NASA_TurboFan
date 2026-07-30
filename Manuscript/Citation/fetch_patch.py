"""
fetch_patch.py
Handles the 10 cases that failed in fetch_citations.py:
  - IEEE DOIs (404 on transform): use CrossRef JSON -> build RIS manually
  - arXiv papers: use SSL-unverified connection
  - ref25 false-positive fix
"""

import os, json, time, ssl
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DELAY   = 0.6
UA      = "CitationFetcher/1.0 (NASA TurboFan RUL research; mailto:spiceyoon@gmail.com)"

# SSL context that skips certificate verification (for arXiv)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE

def get(url, ssl_ctx=None, timeout=30):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl_ctx) as r:
            raw = r.read()
            try:    return raw.decode("utf-8")
            except: return raw.decode("latin-1")
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception as e:
        return None, str(e)

def save(filename, content):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ── CrossRef JSON -> RIS ─────────────────────────────────────────────────────

def crossref_json_to_ris(doi):
    enc = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{enc}"
    raw = get(url)
    if isinstance(raw, tuple) or not raw:
        return None, f"HTTP {raw[1] if isinstance(raw, tuple) else 'error'}"
    try:
        data   = json.loads(raw).get("message", {})
        ty     = data.get("type", "journal-article")
        type_map = {
            "journal-article": "JOUR",
            "proceedings-article": "CONF",
            "book-chapter": "CHAP",
            "posted-content": "UNPB",
            "monograph": "BOOK",
        }
        ris_type = type_map.get(ty, "GEN")

        lines = [f"TY  - {ris_type}"]

        for auth in data.get("author", []):
            fam  = auth.get("family", "")
            giv  = auth.get("given", "")
            name = f"{fam}, {giv}".strip(", ")
            if name:
                lines.append(f"AU  - {name}")

        title_list = data.get("title", [])
        if title_list:
            lines.append(f"TI  - {title_list[0]}")

        journal = data.get("container-title", [])
        if journal:
            lines.append(f"JO  - {journal[0]}")

        pub = data.get("published", {}).get("date-parts", [[None]])[0]
        if pub and pub[0]:
            lines.append(f"PY  - {pub[0]}")

        vol = data.get("volume", "")
        if vol: lines.append(f"VL  - {vol}")

        iss = data.get("issue", "")
        if iss: lines.append(f"IS  - {iss}")

        page = data.get("page", "")
        if "-" in str(page):
            sp, ep = page.split("-", 1)
            lines += [f"SP  - {sp.strip()}", f"EP  - {ep.strip()}"]
        elif page:
            lines.append(f"SP  - {page}")

        publisher = data.get("publisher", "")
        if publisher: lines.append(f"PB  - {publisher}")

        lines.append(f"DO  - {doi}")
        lines.append(f"UR  - https://doi.org/{doi}")
        lines.append("ER  - ")
        return "\n".join(lines), "OK (JSON->RIS)"
    except Exception as e:
        return None, str(e)

# ── arXiv API -> RIS  (SSL bypass) ──────────────────────────────────────────

def arxiv_ris(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    raw = get(url, ssl_ctx=SSL_CTX)
    if isinstance(raw, tuple) or not raw:
        return None, f"HTTP {raw[1] if isinstance(raw, tuple) else 'error'}"
    try:
        root  = ET.fromstring(raw)
        ns    = {"a": "http://www.w3.org/2005/Atom",
                 "x": "http://arxiv.org/schemas/atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None, "No entry in arXiv response"

        title   = (entry.findtext("a:title", namespaces=ns) or "").strip().replace("\n"," ")
        year    = (entry.findtext("a:published", namespaces=ns) or "")[:4]
        summary = (entry.findtext("a:summary", namespaces=ns) or "").strip().replace("\n"," ")[:600]

        au_lines = []
        for author in entry.findall("a:author", ns):
            name  = (author.findtext("a:name", namespaces=ns) or "").strip()
            parts = name.split()
            au_lines.append(
                f"AU  - {parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2
                else f"AU  - {name}"
            )

        doi_el   = entry.find("x:doi", ns)
        doi_line = f"DO  - {doi_el.text.strip()}" if (doi_el is not None and doi_el.text) else ""

        lines = ["TY  - UNPB", *au_lines, f"TI  - {title}", f"PY  - {year}"]
        if doi_line: lines.append(doi_line)
        lines += [f"AN  - arXiv:{arxiv_id}",
                  f"UR  - https://arxiv.org/abs/{arxiv_id}",
                  f"AB  - {summary}", "ER  - "]
        return "\n".join(lines), "OK (arXiv+SSL)"
    except Exception as e:
        return None, str(e)

# ── patch list ───────────────────────────────────────────────────────────────

PATCHES = [
    # IEEE papers: use JSON->RIS
    ("09", "doi-json", "10.1109/TIM.2022.3163761",  "ref09_jin2022.ris"),
    ("12", "doi-json", "10.1109/JSEN.2023.3243540", "ref12_li2023.ris"),
    ("13", "doi-json", "10.1109/TIM.2023.3312337",  "ref13_wang2023.ris"),
    ("14", "doi-json", "10.1109/JSEN.2023.3335994", "ref14_you2024.ris"),
    # arXiv papers: SSL bypass
    ("22", "arxiv",    "2604.13459",  "ref22_abdullah2026.ris"),
    ("23", "arxiv",    "2110.02454",  "ref23_kim2022_revin.ris"),
    ("26", "arxiv",    "2510.04667",  "ref26_noise_signal2025.ris"),
    ("27", "arxiv",    "2601.10269",  "ref27_early_fault2026.ris"),
    ("34", "arxiv",    "2602.19263",  "ref34_bayesian_np2026.ris"),
    ("41", "arxiv",    "2010.09964",  "ref41_chung2021.ris"),
    # ref25: previous hit was wrong paper; try known arXiv ID
    ("25", "arxiv",    "2503.18498",  "ref25_berthelier2026.ris"),
]

results = {}

for refnum, strategy, identifier, filename in PATCHES:
    print(f"[ref{refnum}] {filename} ...", end=" ", flush=True)

    if strategy == "doi-json":
        content, msg = crossref_json_to_ris(identifier)
    elif strategy == "arxiv":
        content, msg = arxiv_ris(identifier)
    else:
        content, msg = None, "unknown strategy"

    if content:
        save(filename, content)
        ok = True
    else:
        ok = False

    print(msg)
    results[refnum] = (ok, msg, filename, identifier)
    time.sleep(DELAY)

# ── summary ──────────────────────────────────────────────────────────────────
succeeded = [(r, d) for r, (ok, _, fn, d) in results.items() if ok]
failed    = [(r, d, m) for r, (ok, m, fn, d) in results.items() if not ok]

print(f"\n{'='*60}")
print(f"PATCH SUCCESS ({len(succeeded)}/{len(PATCHES)})")
for r, d in succeeded:
    print(f"  ref{r}: {d}")
if failed:
    print(f"\nPATCH STILL FAILED ({len(failed)}):")
    for r, d, m in failed:
        print(f"  ref{r}: {d} -> {m}")
