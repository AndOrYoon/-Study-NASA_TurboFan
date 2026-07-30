# Manuscript Guideline
**Paper:** From Fleet Normalization to Fault-Mode Gating: A Cross-Dataset Ablation Study of Turbofan RUL Prediction
**Target venues:** IEEE Transactions on Industrial Informatics (TII) · IEEE (General Transactions/Conference) · Procedia Computer Science (Elsevier)
**Compiled:** 2026-07-03 · Sources: IEEE Author Center, Elsevier Guide for Authors, COPE, OSF, NeurIPS Checklist

---

## 1. Format Comparison — IEEE vs. Procedia CS

| Item | IEEE (Journal/Conference) | Procedia Computer Science |
|------|--------------------------|--------------------------|
| **Page size** | A4 or US Letter | A4 (trim: 192×262 mm) |
| **Layout** | Two-column | **Single-column** |
| **Body font** | Times New Roman 10 pt | Times New Roman 10 pt |
| **Title font** | Times New Roman 24 pt, centered | Times New Roman **bold 17 pt** |
| **Section heading** | Roman numerals, ALL CAPS (`I. INTRODUCTION`) | Arabic numerals, bold 10 pt (`1. Introduction`) |
| **Subsection heading** | Letter + italic (`A. Subsection`) | Arabic decimal, italic 10 pt (`1.1 Subsection`) |
| **Caption font** | ~8–10 pt below figure / above table | 8 pt, left-justified |
| **Page limit (journal)** | 8–10 pp (IEEE Access: no limit) | Set by host conference (typically **≤10 pp**) |
| **Page limit (conference)** | 4–6 pp (varies: VIS=9+2, CAI=6+2) | Set by host conference |
| **Abstract limit** | **150–250 words**, single paragraph | **≤250 words** |
| **Keywords** | 3–4 phrases, alphabetical order | **1–7** phrases, semicolon-separated |
| **Figure caption position** | Below figure (`Fig. 1.`) | Below figure |
| **Table caption position** | **Above table** (`TABLE I`) | Above table |
| **Equation numbers** | Right-aligned parentheses: `(1)` | Right-aligned parentheses: `(1)` |
| **Figure resolution** | 300 dpi (color/gray) / 600 dpi (line art) | **≥300 dpi** |
| **Template** | Official `.cls` + `.docx` at ieeeauthorcenter.ieee.org | `elsarticle` + `ecrc.sty` or `.dot` Word template |
| **Overleaf** | Available | Available (conference-specific) |

### Mandatory Section Order

**IEEE:**
Abstract → Index Terms → I. Introduction → (Related Work) → (Methodology) → (Results) → (Conclusion) → References → Appendix → Author Biographies

**Procedia CS:**
Title → Authors → Affiliations → Abstract → Keywords → Main text → Acknowledgements → References → Appendix

---

## 2. Reference Style

### IEEE (Numbered, Sequential)

- Numbered **in order of first citation** in text — not alphabetical
- In-text: `[1]`, `[2]`, `[1]–[3]` — square brackets, space before bracket, inside punctuation
- Author format: `J. Smith` (initial + last name; NOT `Smith, J.`)
- Use `et al.` only when original source omits names
- Include DOI at end: `doi: 10.xxxx/xxxxxx`

**Journal article:**
```
R. Jin et al., "Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction,"
IEEE Trans. Instrum. Meas., vol. 71, 2022, doi: 10.1109/TIM.2022.3163761.
```

**Conference paper:**
```
S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long Short-Term Memory Network for Remaining
Useful Life Estimation," in Proc. IEEE ICPHM, 2017, pp. 88–95, doi: 10.1109/ICPHM.2017.7998311.
```

**arXiv preprint:**
```
M. E. B. Abdullah, "Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL
Prediction," arXiv:2604.13459, Apr. 2026. [Online]. Available: https://arxiv.org/abs/2604.13459
```

### Procedia Computer Science (Vancouver Numbered)

- Same numbered style as IEEE; BibTeX: `elsarticle-num.bst`
- In-text: `[1]`, `[1,3]`, `[1–5]`
- Author format: `Surname A, Surname B` (last name then initials)

**Journal article:**
```
[1] Jin R, Chen Z, Qi Y, Yin Z. Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful
Life Prediction. IEEE Trans Instrum Meas 2022;71. doi:10.1109/TIM.2022.3163761.
```

> **Key difference:** IEEE uses initials-first; Procedia/Elsevier uses surname-first. Check the BibTeX `.bst` file — it reformats automatically.

---

## 3. Ethics Policies

### 3.1 Duplicate/Concurrent Submission

| Rule | IEEE | Procedia CS (Elsevier) |
|------|------|----------------------|
| Concurrent submission to two venues | **Prohibited** | **Prohibited** |
| Evolutionary publication (workshop → conference → journal) | Allowed — must cite prior versions and explain new contribution | Allowed — must cite prior versions |
| Posting to arXiv before submission | Generally acceptable; disclose preprint | Acceptable (preprints excluded from duplicate submission rule) |

### 3.2 Plagiarism & Self-Plagiarism

- **IEEE:** Checked automatically via CrossRef iThenticate on all submissions. Recycling own text without disclosure = self-plagiarism. Consequence: permanent loss of IEEE publication privileges.
- **Procedia CS:** Elsevier uses iThenticate. "Plagiarism in all its forms constitutes unethical behaviour and is unacceptable."
- **Self-plagiarism threshold:** Identical or substantively equivalent text and content. Mark reused portions clearly, cite original, explain differences.

### 3.3 Authorship Criteria

Both IEEE and Elsevier require **ALL THREE** of the following:

1. **Significant intellectual contribution** to conception, design, execution, or interpretation
2. **Contributed to drafting** or critically revising for intellectual content
3. **Approved the final version** for publication

Contributors not meeting all three → Acknowledgements only, not authors.

**AI tools (ChatGPT, Claude, etc.) cannot be listed as authors under any circumstance** — COPE rule, universally enforced by IEEE, Elsevier, Springer, Wiley, and all major publishers as of 2023.

### 3.4 Conflict of Interest

- **IEEE:** Disclose financial relationships, funding sources, and sponsor involvement in study design.
- **Procedia CS:** Declare all financial/personal relationships that could bias the work; all funding sources.

---

## 4. AI / LLM Usage Policy

### 4.1 IEEE Policy (2024–2026)

> **Mandatory disclosure** in the **Acknowledgments section** whenever AI-generated content appears in the manuscript.

Required elements:
- Name the specific AI system (e.g., "Claude 3.5 Sonnet, Anthropic")
- Identify which sections contain AI-generated or AI-assisted content
- Include a citation to the AI system used

**Prohibited:**
- Listing LLM as author
- Generating substantive text without disclosure
- Reviewers using AI tools with uploaded manuscript text (confidentiality breach)

**Exempted (no disclosure needed):**
- Grammar/spelling correction only (Grammarly-level edits)

**Sample Acknowledgment wording (IEEE):**
> "The authors used Claude (Anthropic, 2026) to assist in code development, debugging, and initial drafting of certain sections. All content was reviewed, validated, and substantially revised by the authors, who take full responsibility for the accuracy of this work."

### 4.2 Elsevier / Procedia CS Policy (updated October 2025)

> Mandatory **separate declaration section** at the end of the manuscript, immediately above References.

**Required heading (exact):**
> **Declaration of Generative AI and AI-assisted technologies in the writing process**

**Template wording:**
> "During the preparation of this work the author(s) used [NAME TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication."

**Prohibited:**
- AI authorship
- Creating or altering figures with generative AI (unless AI generation is the subject of the study)
- Uploading manuscript to AI tools that train on inputs during peer review

**Exempted:** Basic grammar/spelling correction tools.

### 4.3 COPE Universal Rules (applies to all venues)

| Rule | Status |
|------|--------|
| AI as author | Universally **prohibited** (cannot take responsibility, assert conflicts, manage IP) |
| Disclosure of LLM use in writing | **Required** by IEEE, Elsevier, Springer, Wiley, T&F, SAGE |
| Disclosure of AI-assisted data analysis / code | **Expected** (emerging consensus 2024–2025; disclose in Methods) |
| AI-generated figures | **Prohibited** unless AI is the subject of research |
| Grammar-only AI assistance | Generally **exempt** from disclosure |

---

## 5. Open Science Framework (OSF) — Research Ethics

### 5.1 Preregistration

**What it is:** Time-stamped, immutable public declaration of hypotheses and analysis plan submitted *before* data collection.

**Status for this paper:** Hypotheses H2–H7 were formulated before experiments but experiments are now complete → **prospective preregistration is no longer possible**.

**Recommended disclosure:**
> "Research hypotheses were formulated prior to conducting experiments, as documented in pre-experimental design files available at [OSF link]. However, formal prospective preregistration was not completed; all findings should be interpreted accordingly."

Consider depositing `Hypothesis/Hypothesis_PossibleValidation.MD` and `Data_Analysis/Analysis_Plan.md` to OSF as evidence of prior commitment.

### 5.2 Open Data & Open Materials Badges

| Badge | Requirement | Action for This Paper |
|-------|-------------|----------------------|
| **Open Data** | All reproduction-needed data + codebook publicly available with DOI | Note: NASA CMAPSS raw data is already public. Deposit derived feature files (residualized splits per seed) to OSF/Zenodo |
| **Open Materials** | All scripts/code/analysis materials publicly available | Upload `Data_Analysis/Code/` (H2–H7 + shared utils) + `requirements.txt` to GitHub + Zenodo DOI |
| **Preregistered** | Study registered before data collection | Not claimable (post-hoc); deposit design docs as supporting evidence |

**Repository requirements:** OSF, Zenodo, Figshare, or re3data-verified repository. Must have a **persistent DOI** and open license (CC BY recommended).

### 5.3 Reproducibility Checklist (NeurIPS/ICML Standard — expected by IEEE)

All of the following **must be reported** in the Methods or Supplementary:

- [ ] **Random seeds:** `[0, 1, 2, 3, 4]` — state explicitly
- [ ] **Number of runs:** 5 per condition — state explicitly
- [ ] **All hyperparameters:** LSTM hidden=64, dropout=0.2, window=30, batch=256, LR=1e-3, WD=1e-4, patience=15 — report in a table
- [ ] **How hyperparameters were chosen:** prior literature (Zheng 2017 + Li 2018) and pilot experiments
- [ ] **Error bars:** `mean ± std` across 5 seeds — already done ✓
- [ ] **What error bars represent:** variance across 5 fixed seeds, same engine-level train/val/test split — state this
- [ ] **Hardware:** GPU type, total compute hours
- [ ] **Software versions:** Python 3.13, PyTorch version, scikit-learn version
- [ ] **Statistical tests:** Wilcoxon rank-sum (one-sided, α=0.05) + Benjamini-Hochberg FDR correction — already done ✓
- [ ] **Code availability link:** GitHub/Zenodo DOI

### 5.4 Pre-registration & P-hacking Prevention

This paper already uses the correct safeguards:
- ✅ Wilcoxon + BH-FDR correction (prevents false positives from multiple comparisons)
- ✅ 5 fixed seeds (prevents cherry-picking)
- ✅ Pre-specified hypotheses (H2–H7 documented before experiments in `Hypothesis/`)
- ✅ Engine-level val split (prevents test-set contamination)
- ✅ All loss functions × all datasets evaluated (no selective reporting)

**Remaining disclosure needed:**
- The paper must distinguish: (a) *confirmatory* results matching pre-specified hypotheses vs. (b) *exploratory* sub-findings observed after experiments.
- Label exploratory findings explicitly: "This finding was not pre-specified; it emerged from post-hoc analysis of…"

---

## 6. Actionable Pre-Submission Checklist

### Manuscript Content

- [ ] Abstract ≤250 words (current draft: ~235 words ✓)
- [ ] 3–7 keywords listed (use IEEE Thesaurus for IEEE submission)
- [ ] All section headings match venue style (Roman numerals for IEEE; numbered for Procedia)
- [ ] All abbreviations defined at first mention
- [ ] All equations numbered with right-aligned parentheses
- [ ] Figure captions below figure; table captions above table
- [ ] References in IEEE or Vancouver numbered format (depending on venue)
- [ ] DOIs included for all references where available
- [ ] arXiv preprints cited with arXiv ID and access date

### Ethics & Disclosure

- [ ] **AI disclosure statement** added (Acknowledgments for IEEE; separate section for Procedia)
- [ ] No AI tool listed as author
- [ ] Conflict of interest statement added
- [ ] Funding sources declared
- [ ] Authorship criteria verified for all co-authors
- [ ] No duplicate submission confirmed

### Open Science

- [ ] Code uploaded to GitHub + Zenodo DOI obtained
- [ ] `requirements.txt` with pinned versions created
- [ ] Data availability statement added to manuscript
- [ ] Pre-experiment design documents deposited to OSF
- [ ] Hyperparameter table added to Methods section
- [ ] Compute resources described (GPU, total run time)
- [ ] Statistical testing procedure described (Wilcoxon + BH-FDR + Cohen's d)

---

## 7. Key Source URLs

| Resource | URL |
|----------|-----|
| IEEE Author Center | https://ieeeauthorcenter.ieee.org/ |
| IEEE AI/LLM Policy | https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/ |
| IEEE Reference Guide | https://journals.ieeeauthorcenter.ieee.org/ (Editorial Style Manual PDF) |
| IEEE Templates | https://www.ieee.org/conferences/publishing/templates.html |
| Procedia CS Guide for Authors | https://www.sciencedirect.com/journal/procedia-computer-science/publish/guide-for-authors |
| Elsevier AI Policy | https://www.elsevier.com/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-writing-for-elsevier |
| Elsevier Ethics Policy | https://www.elsevier.com/about/policies-and-standards/publishing-ethics |
| COPE AI Authorship Statement | https://publicationethics.org/guidance/cope-position/authorship-and-ai-tools |
| OSF Preregistration | https://osf.io/prereg/ |
| OSF Open Badges Wiki | https://osf.io/tvyxz/wiki |
| Zenodo (DOI repository) | https://zenodo.org/ |
| NeurIPS Paper Checklist | https://neurips.cc/public/guides/PaperChecklist |
