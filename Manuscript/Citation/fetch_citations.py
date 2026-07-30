"""
fetch_citations.py
Download RIS/ENW citation files for all 42 references in References.md.
Sources: CrossRef API (DOI-based), arXiv API (arXiv-only papers),
         CrossRef title search (unknown DOI papers).
Output: one .ris file per reference in the same directory as this script.
"""

import os, time, json
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DELAY   = 0.6   # seconds between requests (CrossRef polite pool)

UA = "CitationFetcher/1.0 (NASA TurboFan RUL research; mailto:spiceyoon@gmail.com)"

# ── helpers ──────────────────────────────────────────────────────────────────

def get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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

def crossref_ris(doi):
    """Download RIS from CrossRef transform endpoint."""
    enc = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{enc}/transform/application/x-research-info-systems"
    result = get(url)
    if isinstance(result, tuple): return None, result[1]
    if result and result.strip().startswith("TY"): return result, "OK"
    return None, f"Unexpected content: {repr(result[:80]) if result else 'None'}"

def crossref_title_search(title):
    """Search CrossRef by title, return best-match DOI or None."""
    q   = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=1&select=DOI,title,author,published"
    result = get(url)
    if isinstance(result, tuple) or not result: return None
    try:
        data  = json.loads(result)
        items = data.get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    except Exception:
        pass
    return None

def arxiv_ris(arxiv_id):
    """Download arXiv metadata via API and convert to RIS."""
    url    = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    result = get(url)
    if isinstance(result, tuple) or not result:
        return None, result[1] if isinstance(result, tuple) else "No response"
    try:
        root  = ET.fromstring(result)
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
            if len(parts) >= 2:
                au_lines.append(f"AU  - {parts[-1]}, {' '.join(parts[:-1])}")
            else:
                au_lines.append(f"AU  - {name}")

        # Check for DOI
        doi_el = entry.find("x:doi", ns)
        doi_line = f"DO  - {doi_el.text.strip()}" if doi_el is not None and doi_el.text else ""

        lines = ["TY  - UNPB", *au_lines, f"TI  - {title}", f"PY  - {year}"]
        if doi_line:
            lines.append(doi_line)
        lines += [f"AN  - arXiv:{arxiv_id}",
                  f"UR  - https://arxiv.org/abs/{arxiv_id}",
                  f"AB  - {summary}", "ER  - "]
        return "\n".join(lines), "OK (arXiv)"
    except Exception as e:
        return None, str(e)

# ── paper registry ────────────────────────────────────────────────────────────
# Each entry: (ref_num, strategy, identifier, filename, fallback_title)
# strategy: "doi" | "arxiv" | "search"

PAPERS = [
    # ── Background / Foundation ──────────────────────────────────────────────
    ("01", "search", None,
     "ref01_saxena2008.ris",
     "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"),

    ("02", "doi",   "10.1109/PHM.2008.4711422",
     "ref02_heimes2008.ris", None),

    ("03", "doi",   "10.36001/ijphm.2014.v5i2.2236",
     "ref03_ramasso2014.ris", None),

    ("04", "doi",   "10.1007/978-3-319-32025-0_14",
     "ref04_babu2016.ris", None),

    ("05", "doi",   "10.1109/ICPHM.2017.7998311",
     "ref05_zheng2017.ris", None),

    ("06", "doi",   "10.1016/j.ress.2017.11.021",
     "ref06_li2018.ris", None),

    ("07", "doi",   "10.1016/j.ress.2019.01.016",
     "ref07_ellefsen2019.ris", None),

    ("08", "doi",   "10.1007/s10845-021-01750-x",
     "ref08_mo2021.ris", None),

    ("09", "doi",   "10.1109/TIM.2022.3163761",
     "ref09_jin2022.ris", None),

    ("10", "doi",   "10.1016/j.ress.2022.108648",
     "ref10_xu2022.ris", None),

    ("11", "doi",   "10.1016/j.ress.2023.109258",
     "ref11_zhang2023.ris", None),

    ("12", "doi",   "10.1109/JSEN.2023.3243540",
     "ref12_li2023.ris", None),

    ("13", "doi",   "10.1109/TIM.2023.3312337",
     "ref13_wang2023.ris", None),

    ("14", "doi",   "10.1109/JSEN.2023.3335994",
     "ref14_you2024.ris", None),

    ("15", "doi",   "10.1038/s41598-025-09155-z",
     "ref15_elsherif2025.ris", None),

    ("16", "doi",   "10.3390/s24082543",
     "ref16_wu2024.ris", None),

    ("17", "doi",   "10.1016/j.ymssp.2024.111120",
     "ref17_li2024.ris", None),

    # ── H2 Clipping ──────────────────────────────────────────────────────────
    ("18", "doi",   "10.1109/ICTAI66417.2025.00160",
     "ref18_imbert2026.ris", None),

    ("19", "doi",   "10.3390/app132111893",
     "ref19_ensarioglu2023.ris", None),

    ("20", "doi",   "10.36001/phmap.2023.v4i1.3611",
     "ref20_srinivasan2023.ris", None),

    ("21", "doi",   "10.1016/j.conengprac.2023.105840",
     "ref21_arunan2024.ris", None),

    ("22", "arxiv", "2604.13459",
     "ref22_abdullah2026.ris", None),

    # ── H5 Normalization ─────────────────────────────────────────────────────
    ("23", "arxiv", "2110.02454",
     "ref23_kim2022_revin.ris", None),

    ("24", "search", None,
     "ref24_zhang2022_moc.ris",
     "A Framework for Predicting the Remaining Useful Life of Machinery Working under Time-Varying Operational Conditions"),

    ("25", "search", None,
     "ref25_berthelier2026.ris",
     "On the Role of Reversible Instance Normalization"),

    ("26", "arxiv", "2510.04667",
     "ref26_noise_signal2025.ris", None),

    ("27", "arxiv", "2601.10269",
     "ref27_early_fault2026.ris", None),

    ("28", "doi",   "10.1145/3690624.3709260",
     "ref28_inflow2025.ris", None),

    ("29", "doi",   "10.1007/s44196-024-00639-w",
     "ref29_deng2024.ris", None),

    # ── H6 Fault Mode ────────────────────────────────────────────────────────
    ("30", "doi",   "10.1016/j.aei.2024.102360",
     "ref30_multitask2024.ris", None),

    ("31", "doi",   "10.3390/app15147884",
     "ref31_gmm_lstm2025.ris", None),

    ("32", "doi",   "10.3390/e27010079",
     "ref32_moeformer2025.ris", None),

    ("33", "doi",   "10.3390/aerospace11040293",
     "ref33_fcamslstm2024.ris", None),

    ("34", "arxiv", "2602.19263",
     "ref34_bayesian_np2026.ris", None),

    ("35", "doi",   "10.36001/ijphm.2023.v14i2.3486",
     "ref35_fault_prog2023.ris", None),

    ("36", "doi",   "10.1038/s41598-025-23473-2",
     "ref36_ozcan2025.ris", None),

    # ── H7 Loss Functions ────────────────────────────────────────────────────
    ("37", "doi",   "10.3390/s20030723",
     "ref37_rengasamy2020a.ris", None),

    ("38", "search", None,
     "ref38_rengasamy2020b.ris",
     "Asymmetric Loss Functions for Deep Learning Early Predictions of Remaining Useful Life in Aerospace Gas Turbine Engines"),

    ("39", "search", None,
     "ref39_liu2021.ris",
     "A Multi-Head Neural Network with Unsymmetrical Constraints for Remaining Useful Life Prediction"),

    ("40", "doi",   "10.3390/s26072249",
     "ref40_diao2026.ris", None),

    ("41", "arxiv", "2010.09964",
     "ref41_chung2021.ris", None),

    ("42", "search", None,
     "ref42_asif2022.ris",
     "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset"),
]

# ── main loop ─────────────────────────────────────────────────────────────────

results = {}  # refnum -> (ok, msg, filename, identifier_used)

for entry in PAPERS:
    refnum, strategy, identifier, filename, title = entry
    print(f"[ref{refnum}] {filename} ...", end=" ", flush=True)

    ok, msg = False, "not attempted"

    if strategy == "doi":
        content, msg = crossref_ris(identifier)
        if content:
            save(filename, content)
            ok = True
        time.sleep(DELAY)

    elif strategy == "arxiv":
        content, msg = arxiv_ris(identifier)
        if content:
            save(filename, content)
            ok = True
        time.sleep(DELAY)

    elif strategy == "search":
        # Try CrossRef title search to find DOI, then download
        found_doi = crossref_title_search(title)
        time.sleep(DELAY)
        if found_doi:
            content, msg2 = crossref_ris(found_doi)
            time.sleep(DELAY)
            if content:
                save(filename, content)
                ok  = True
                msg = f"OK via title search → DOI:{found_doi}"
            else:
                msg = f"Found DOI {found_doi} but CrossRef returned: {msg2}"
        else:
            msg = "Title search returned no DOI"

    print(msg)
    results[refnum] = (ok, msg, filename, identifier or f"title:{title[:40]}")

# ── summary ──────────────────────────────────────────────────────────────────
succeeded = [(r, d) for r, (ok, _, fn, d) in results.items() if ok]
failed    = [(r, d, m) for r, (ok, m, fn, d) in results.items() if not ok]

print(f"\n{'='*60}")
print(f"SUCCESS ({len(succeeded)}/{len(PAPERS)})")
for r, d in succeeded:
    print(f"  ref{r}: {d}")

print(f"\nFAILED ({len(failed)}/{len(PAPERS)}) — requires manual download:")
for r, d, m in failed:
    print(f"  ref{r}: {d}")
    print(f"         reason: {m}")

# Write summary file
summary_path = os.path.join(OUT_DIR, "download_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Citation download summary\n{'='*60}\n\n")
    f.write(f"SUCCESS ({len(succeeded)}/{len(PAPERS)})\n")
    for r, d in succeeded:
        _, _, fn, _ = results[r]
        f.write(f"  [ref{r}] {fn}  ← {d}\n")
    f.write(f"\nFAILED ({len(failed)}/{len(PAPERS)}) — manual download required\n")
    for r, d, m in failed:
        _, _, fn, _ = results[r]
        f.write(f"  [ref{r}] {fn}\n")
        f.write(f"          identifier: {d}\n")
        f.write(f"          reason: {m}\n\n")

print(f"\nSummary saved → {summary_path}")
