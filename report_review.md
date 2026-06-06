# Report Review — PCN Labs

---

## Global / Structural Issues

- 👍**Title page**: Title reads "Minimal LaTeX Document" and author reads "Your Name" — both are unfilled placeholders.
- **No abstract** anywhere in the document.
- 👍**No table of contents**.
- **No conclusion chapter** — none of the three main chapters have a concluding summary section (Ch.1 has §1.4.5, but Ch.2 ends abruptly after the comparison paragraph and Ch.3 has no conclusion at all).

---

## Chapter 1 — QKD

### §1.1 Introduction

- "in its original **for**, is based on 4 quantum states" → typo, should be "**form**".
- "In this chapter we try to **enact** the first scheme" — odd word choice; use "implement" or "demonstrate".
- The introduction ends mid-sentence at the bottom of the page ("it can encode quantum") and continues with the setup diagram on the next page. The transition is abrupt and the cut-off sentence reads as incomplete regardless of the page break.

---

### §1.2 Implementation of a BB84 demonstration device

#### §1.2.2 Experimental Setup
- **Entire section is a placeholder**. The text present is the lab-sheet instruction ("Students should list the materials and instruments used in the experiment..."), not actual student-written content. There is no schematic, no equipment list, no actual setup description.

#### §1.2.3 Results
- **Entire section is a placeholder template**. The text is the original instructions listing what should be reported, including the literal string "o QKD test with Eve" (an unformatted bullet "o"). No actual data is present: no bases tables, no key, no QBER values, neither for the no-Eve nor the with-Eve case.

---

### §1.3 Implementation of the OTP Protocol

#### §1.3.2 Setup description
- `TODO: check the libraries and eventually explain` left in the text.
- `TODO: mettere jupyter` left in the text — also in Italian ("mettere" = "to put").
- `binascii` appears in the library list but is never used or discussed in any of the code listings shown.

#### §1.3.3 Results — Encrypting a word
- "the number of usable key bits **(TODO:plural?)** is 29" — TODO left visible in the text.
- "**TODO**: put reference to where otp is defined" — left in text.

#### §1.3.3 Results — Encrypted communication
- "decoded to ASCII characters, finalizing the decryption. **and and** then decoded" (p.6) — doubled "and".
- "The get\_key\_from\_files simply access the file specified by the **aprticular**" (p.9) — typo, should be "particular".
- **Large all-caps TODO block** at the end of the section (p.9): "TODO: CHECK THE DIMENSIONS AND NAMES. ADD THE FACT THAT THE CODE CAN BE CHECKED LOCALLY AND SHOW THE OUTPUTS." — not removed.
- There are no sample outputs shown for the encrypted communication experiment. The TODO above acknowledges this, but the section currently ends with code only and no demonstrated results.

#### Listing captions — widespread error
Six or more listings share the identical wrong caption **"command on a given key file."**:
- Listing 1.10 (`xor_bytes` function)
- Listing 1.12 (`send_framed` function)
- Listing 1.13 (`recv_framed` function)
- Listing 1.14 (`recv_exact` function)
- Listing 1.16 (`get_key_from_file` function)
- Listing 1.17 (`decrypt` function)
- Listing 1.18 (`handle_client` function)

Each caption should describe what the respective function does.

---

### §1.4 Characterization of QKD components

#### §1.4.1 Objective
- "**Which** is a radiometric measurement principle used to determine..." — standalone sentence starting with "Which"; should be a relative clause attached to the preceding sentence, not a new sentence.

#### §1.4.2 Components / Setup
- Table 1.2 caption missing a full stop at the end.
- The section introduces the detector as "**SPD**" (single-photon detector) throughout, but the Results subsection (§1.4.4) switches to "**SPAD**" (single-photon avalanche diode) without prior introduction or reconciliation. Use one term consistently, or introduce the more specific SPAD term explicitly when first used.

#### §1.4.4 Results
- Table 1.3 caption missing a full stop at the end.
- "The results are summarized in **Table 1.4.4**." — this appears to be a broken cross-reference (`\ref{}` resolving to a section number instead of a table number). The correct reference is likely Table 1.3.
- The background count rate description "(measured 10 times and averaged)" is parenthetical inline text; it should be stated more formally, e.g., "averaged over 10 measurements".

#### §1.4.5 Conclusion
- The conclusion correctly identifies the large discrepancy in η. Consider adding a sentence on what improvement would fix the calibration (e.g., better attenuation characterisation of the SPAD arm) to make the section feel complete rather than just identifying the problem.

---

## Chapter 2 — Atomic Spectroscopy

### §2.2 Laser Characterization

#### §2.2.1 Results and Analysis
- Figures 2.1 and 2.2 are rendered very small and side-by-side. The axis labels and tick values are completely unreadable at print size. These should be larger or shown individually.
- Third bullet point: "**between** 56.5 mA and 60 mA, the output power rises..." — should be capitalised ("**Between**") as it opens a bullet item.
- The threshold is reported two different ways in the same section: I\_{th,fit} = 51.72 mA in the equation, and "≈ 51.7 mA" in prose — consistent but the rounding differs. Use one precision throughout.

---

### §2.3 Atomic Spectroscopy

#### §2.3.2 Data Processing
- Figure 2.8 / 2.9 captions: "Calculated **trsasmittance trough** the cesium vapor cell." — two typos in one caption: "trsasmittance" → "transmittance", "trough" → "through".

#### §2.3.3 Results and Discussion

**Comparison section (p.30):**
- "**THIS MAKE SENSE FOR US?**" — left as a question inside a bullet point in the stray light discussion. Must be resolved and removed.
- "**vapour**" used in the Comparison section (p.30); "**vapor**" used everywhere else in the chapter (e.g., p.20, p.22). Pick one spelling and apply it consistently throughout.

**Notation issues:**
- Results n = 1.33 ± 0.13 × 10¹⁶ m⁻³ and n\_{th} = 3.99 ± 0.52 × 10¹⁶ m⁻³ are ambiguous: operator precedence makes them read as 1.33 ± (0.13 × 10¹⁶). Should be written **(1.33 ± 0.13) × 10¹⁶ m⁻³** (same issue applies to n\_{th}). This is a standard scientific notation rule.
- Cell length written as L = 0.010(1) m — compact parenthetical uncertainty notation is valid but inconsistent with the ± format used everywhere else in the report. Either use it throughout or convert to L = (10 ± 1) mm.
- In eq. (2.21), Boltzmann's constant (1.38 × 10⁻²³) appears without units; should be J K⁻¹.
- "natural linewidth Γ = 28.743 Mrad/s" — no uncertainty given here (it's taken from the reference [9]), but the quantity is used directly in the density calculation. At minimum, cite that this value is taken as exact from [9] in this context.

---

## Chapter 3 — Clean Room

### §3.1 MICRO Experience: Photolithography

- "Photolithography became **fundemental** due to batch fabrication" (p.32) — typo, should be "**fundamental**".
- Figure 3.1 includes embedded annotation text from the original manufacturer figure; the caption style does not match the rest of the report.
- The profilometer step-height measurement is mentioned in §3.1.2 ("Resist thickness and step height were subsequently measured with a **profilometer**") but no numerical result is ever reported. The data is missing.

---

### §3.2 NANO Experience: Focused Ion Beam

#### §3.2.2 Results
- **Entirely empty** — the heading "3.2.2 Results" appears with no content before §3.3 begins. All results from the EBL/lift-off process are missing.

---

### §3.3 Dry-Etch and SEM Imaging

This is the most incomplete section of the entire report.

#### §3.3.1 Materials and Methods — Setup description
- "**TODO** Below, a complete list of machines and instruments used to carry the activities is presented." — TODO placeholder, no list follows.
- "Box cutter (**TODO** I do not remember if it was special or not, like with **diamonds stones**.);" — TODO left in; also "diamonds stones" is ungrammatical (should be "diamond-tipped" or similar).
- "2 μm of SiO₂ to create a substrate for etching **(TODO check ??)**" — incomplete, unresolved.
- "**MAchnie** for etching?? TODO put photo." — typo ("MAchnie" → "Machine"), double question marks, TODO left in.

#### §3.3.1 Process flow description
- "**Wafer cutting**: Firstly, both the wafers were cut **(TODO we sure were both?)** with a box cutter. Notice that the **nanostr ??** TODO: perhaps put a schematics..." — multiple TODOs, incomplete fragment "nanostr".
- "the **cutted** structures were glued on a bigger one" — "cutted" is not standard English; should be "cut".
- "**DRIE machine cleaning**: ... TODO: specify the cleaning recipe." — left unresolved.
- "**Pseudo Bosch process**: ... **(TODO vero?)**" — Italian word "vero" (= "right?") left in the text.
- "13.53 MHz **(TODO check 13.56 or 13.53)**" — unresolved factual question (the standard RF frequency is 13.56 MHz).
- "**(TODO:picture)**" in the middle of the RIE description.
- "**(TODO i'm not sure it is effectively less)**" — uncertainty about a claimed fact, left unresolved.
- "**TODO cite temperature importance? TODO:cite void importance made by machine**" — two adjacent TODOs.
- "i **dont** think we did the KOH polishing" — contraction without apostrophe; the entire line is a TODO that should be removed.
- Table 3.4: Duration for Step 3 "Isotropic etch" reads "**TODO?? s**" — incomplete.
- "10 units of silicon are removed **perr** unit of photoresist" — typo ("perr" → "per").

#### §3.3.2 Results
- "**TODO: connect the images here to the corresponding general wafer before.**" — left in opening line.
- "a piece coming from the microstructures' wafer **(TODO: put reference)**" — TODO left in.
- "the negative one couldn't be fabricated due to **TODO??** TODO is there somewhere the description..." — incomplete sentence with stacked TODOs.
- The **"Nanostructure"** subsection has no explanatory text at all; only figures follow the heading.
- **Figures 3.12 through 3.21 (all 10 SEM images)**: every caption reads "This is a sample figure caption that describes the image content." — all are placeholder captions. None of the SEM images are identified, described, or discussed.
- There is **no written analysis** of any SEM image: no etch depth measurements, no wall angle assessment, no comparison between microstructure and nanostructure results, no discussion of scalloping, sidewall quality, or lift-off success.

---

## Summary of Priority Fixes

| Priority | Issue |
|----------|-------|
| Critical | Title page placeholders (title, author name) |
| Critical | §1.2.2 Experimental Setup — entirely placeholder text |
| Critical | §1.2.3 Results — entirely placeholder template, no actual QKD data |
| Critical | §3.2.2 Results — completely empty |
| Critical | §3.3.2 — all 10 SEM captions are placeholders; no analysis written |
| Critical | §3.3.1 — massive TODO density, multiple incomplete sentences, Italian left in |
| High | "THIS MAKE SENSE FOR US?" left in §2.3.3 |
| High | All-caps TODO block at end of §1.3.3 |
| High | 7+ listing captions all read "command on a given key file" |
| High | Profilometer numerical result missing from §3.1.2 |
| High | Broken cross-reference "Table 1.4.4" in §1.4.4 |
| Medium | Ambiguous notation (1.33 ± 0.13) × 10¹⁶ — parentheses needed |
| Medium | "vapour" / "vapor" inconsistency in Ch.2 |
| Medium | Figures 2.1/2.2 unreadably small |
| Medium | "cutted", "aprticular", "fundemental", "trsasmittance", "trough" typos |
| Medium | SPD vs SPAD terminology inconsistency in Ch.1 |
| Medium | TODOs in Italian (§1.3.2, §3.3.1) |
| Low | Missing full stops on Table 1.2 and 1.3 captions |
| Low | "Which is a radiometric..." — fragment sentence in §1.4.1 |
| Low | No abstract, no conclusion chapter |
