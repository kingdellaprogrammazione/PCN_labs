# Lithography Experiment Comprehensive Notes

## 1. MICRO-Lithography (AZ 5214E Image Reversal)

### 1.1 Material & Substrate
*   **Substrate:** 4-inch (100 mm) Silicon dummy wafer.
    *   **Properties:** P-doped (boron), 525 µm thick, single-side polished.
    *   **Rationale:** Dummy wafers are used for process optimization to save costs.
*   **Adhesion Promoter:** HMDS (Hexamethyldisilazane). 
    *   Used to replace Si-OH groups with hydrophobic trimethylsilyl groups to prevent resist delamination.
*   **Photoresist:** **AZ 5214E** (Image-Reversal Resist).
    *   **Thickness:** ~1.3 µm at 4000 rpm.
    *   **Mode:** Image-Reversal (IR) to create an **undercut profile** for lift-off.

### 1.2 Step-by-Step Process Flow
1.  **Dehydration Bake:** 150 °C for 5 min (removes physisorbed water).
2.  **HMDS Vapour Prime:** 150 °C for ~5 min in a closed oven.
3.  **Spin Coating:**
    *   **Spread:** 500 rpm for 5 s.
    *   **Spin:** 4000 rpm, 500 rpm/s acceleration, 30 s.
4.  **Soft Bake:** 110 °C for 60 s on a hot plate (removes solvents).
5.  **Mask Alignment & Exposure:**
    *   **Mode:** Vacuum (hard) contact to minimize diffraction.
    *   **Time:** 1 s.
    *   **Resolution Limit:** $2b_{min} \approx 1.4 \mu m$.
6.  **Post-Exposure Bake (PEB):** 120 °C for 120 s.
    *   **Crucial Step:** Cross-links exposed regions, making them insoluble.
7.  **Flood Exposure:** UV-LED lamp, 100% power for 40 s (maskless).
    *   Sensitizes originally unexposed regions.
8.  **Development:** AZ 400K : H₂O (1:4) for 60 s.
    *   Followed by 30 s DI water rinse and spin-dry.

### 1.3 Result Verification (MICRO Images & Numbers)
*   **Optical Microscope Checks:**
    *   **Appearance:** Bright (yellow/green) = Resist; Dark = Substrate (Silicon).
    *   **Structures:** Parallel trenches (2, 3, 5 µm), radial patterns, alignment marks, contact pads.
    *   **Criteria:** No "scumming" (residue) in windows; clean lift-off profile.
*   **Profilometer Checks:**
    *   **Step Height:** Should be ~1.3 µm.
    *   **Profile:** Look for verticality and lack of resist reflow.
*   **CD (Critical Dimension) Bias:**
    *   Calculate: $Measured Width - Nominal Width$.
    *   Check for discrepancies caused by proximity effects or over-development.

---

## 2. NANO-Lithography (Focused Ion Beam - FIB)

### 2.1 Material & Substrate
*   **Substrate:** 100 nm thick Si₃N₄ support membrane (TEM membrane).
*   **Resist:** S1813 (Positive resist used here as a FIB-sensitive resist).

### 2.2 FIB Process Details
*   **Ion Source:** Gallium (Ga+).
*   **Parameters:** 30 keV energy, 50 nm spot size.
*   **Mechanism:**
    1.  **Sputtering:** Ions pierce the resist to create the hole.
    2.  **Cross-linking:** Secondary electrons deposit dose laterally, creating a hard "reticulated" shell.
*   **Development:** Acetone bath (dissolves un-cross-linked resist, leaving nano-pillars/walls).

### 2.3 Result Verification (NANO)
*   **Structure:** Array of nanopillars.
*   **Target Diameter:** ~150 nm.
*   **Checklist:** Inspect via SEM for pillar verticality and pitch accuracy.

---

## 3. DRIE (Deep Reactive Ion Etching)

### 3.1 Etching Process: Pseudo-Bosch
*   **Samples:** Both the MICRO (trench) and NANO (pillar) samples are processed here.
*   **Technique:** ICP-DRIE (Inductively Coupled Plasma).
*   **Pseudo-Bosch Method:** Simultaneous flow of SF₆ (etchant) and C₄F₈ (passivant).
    *   **Advantage:** Continuous etch, smoother sidewalls, no "scallops" (essential for nano-scale).
    *   **Trade-off:** Lower etch rate compared to standard Bosch.

### 3.2 ICP-DRIE Parameters
| Parameter | Value |
| :--- | :--- |
| Tchiller | 5 °C |
| ICP RF Power | 2500 W |
| Table RF Power | 50 W (during break/etch phases) |
| Etch Rate | ~6 µm/min |
| Selectivity (Si:Resist) | Target ~1:100 (needs verification) |

---

## 4. Final Inspection & SEM Analysis

### 4.1 SEM Setup (Quanta 3D FEG)
*   **High Voltage:** 20 kV.
*   **Tilt Angle:** 52° (crucial for viewing sidewall profiles and depths).

### 4.2 Measurement & Alignment Checklist
*   **MICRO Sample:**
    *   Measure trench depth and width.
    *   Check for **Sidewall Angle:** Deviation from 90°.
    *   Identify any under-etching at the base.
*   **NANO Sample:**
    *   Pillar diameter (Top vs. Base) - check for tapering.
    *   Pillar height and array periodicity (pitch).
*   **Comparison MICRO vs. NANO:**
    *   **Aspect Ratio:** Higher aspect ratios at the nanoscale are harder due to mask erosion and ion shadowing (ARDE - Aspect Ratio Dependent Etching).
    *   **Surface Roughness:** Evaluate the smoothness of the pseudo-Bosch process vs. standard Bosch scallops.

## 5. Summary Table of Equipment

| Step | Equipment |
| :--- | :--- |
| Coating | Spin Coater (programmable) |
| Bakes | Calibrated Hot Plates (110°C / 120°C) |
| Exposure | Contact Mask Aligner (365 nm / Vacuum mode) |
| IR Flood | UV-LED Flood Lamp (100% power) |
| Nano-Patterning | FIB (Ga+ source, 30 keV) |
| Etching | ICP-DRIE System |
| Inspection (Micro) | Optical Microscope (Bright-field) & Profilometer |
| Inspection (Nano) | SEM (Quanta 3D FEG, 52° tilt) |

---
**Note:** The MICRO sample must be carefully stored after development as it serves as the mask for the DRIE stage.
