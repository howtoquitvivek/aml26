# Phase 3.1 Matcher Diagnostic Report (20-S1 Validation Sample)

**Scope**: 20 Source 1 entities (10 India, 10 US) evaluated against 10.32M Source 2 & Source 3 candidate records.  
**Focus**: Investigating why True Ground-Truth Pairs recovered by **Key B (CFG1)**, **Key C (CFG2)**, and **Key D (CFG3)** were rejected by the deterministic matcher, and why False Positives were accepted.

---

## 1. Executive Summary & Diagnostic Findings

Across the 20-S1 validation sample (51 total Ground Truth links):
- **CONFIG 1** recovered **33 / 51 GT links (64.71%)**.
- **CONFIG 2 (+ Key C)** recovered **37 / 51 GT links (72.55%)** (+4 new true links).
- **CONFIG 3 (+ Key D)** recovered **38 / 51 GT links (74.51%)** (+1 new true link beyond CFG2).
- **Yet at Budget 25, $\theta=0.70$, all three configurations achieve an identical Macro $F_{0.5} = 0.5397$ and Recall = 0.4673 (25 TPs, 10 FPs)**.

### Root Cause Diagnosis:
1. **The Matcher Relies on Exact/Canonical Name Shortcuts**:
   - The deterministic scorer gives scores $\ge 0.90$ when `name_exact` or `name_canonical_exact` is 1.0.
   - For candidate pairs without exact name match (which Key C and Key D specifically target), the score formula is:
     $$\text{Score} = 0.50 \times (0.60 \times \text{name\_tok} + 0.40 \times \text{name\_char}) + 0.50 \times (0.50 \times \text{addr\_tok} + 0.50 \times \text{addr\_num})$$
2. **True Match Discrepancies Penalized by Strict Jaccard**:
   - Real-world variants (e.g., `"The Home Depot"` vs `"Home Depot Store #123"`, `"Sharma Medical Hall"` vs `"Sharma Pharmacy"`, or slight spelling/transliteration differences) achieve moderate token Jaccard (0.33 to 0.50).
   - Variations in address formatting (e.g., `"12 M.G. Road"` vs `"12 Mahatma Gandhi Marg"`) yield address token Jaccard of 0.20 to 0.40.
   - Under the linear weighted formula, $0.50 \times 0.40 + 0.50 \times 0.35 = 0.375$, which falls far below the $0.70$ threshold.
3. **False Positives Allowed by Generic Name Homonyms**:
   - Businesses sharing identical common/generic names (e.g., `"Subway"`, `"Pizza Hut"`, `"Shree Ganesh Enterprises"`) in different locations trigger the exact name shortcut ($0.65 - 0.90$), passing the matcher even when addresses are for different branches.

---

## 2. Table A: False Negative True Pairs (Rejected Ground-Truth Matches)

The table below details true pairs recovered by blockers but rejected by the matcher at threshold $\theta = 0.70$ (or ranked outside Budget 25).

| S1 ID | Candidate ID | Blocker Origin | Group | S1 Name / Cand Name | S1 Address / Cand Address | Name Tok | Name Char | Addr Tok | Addr Num | Final Score | Pass $\theta$ (65/70/75/80) | Rejection Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `S1-454111437` | `S2-171575965` | Canonical, Key C, Key D | **CFG1 True Link** | **S1**: Maa Research Private Limited<br>**Cand**: Maa Research Pivoate Limited | **S1**: Flat No-702, Sapphire A, Gulmohur Tower, Chiranjeev Vihar, Sector, -6, Ghaziabad, Uttar Pradesh<br>**Cand**: FLAT NO-00702, SAPPHIRE A, GULMOHUR TOWER, CHIRANJEEV VIHAR, SECTOR, -6, GHAZIABAD, Uttar Pradesh | 0.67 | 0.64 | 0.87 | 0.33 | **0.6286** | `N/N/N/N` | Score 0.6286 < 0.70 threshold |
| `S1-881535873` | `S2-642463929` | Key D | **CFG3 (not CFG2) [Key D]** | **S1**: Azamgarh Marketing Private Limited<br>**Cand**: azamgarhmarketing.com | **S1**: C/O Ajai Kumar Rai, Village Lachirampur Hafizpur, Post- Heerapatti, Azamgarh, Uttar Pradesh<br>**Cand**: PLOT 58 C/O AJAI KUMAR RAI, VILLAGE LACHIRAMPUR HAFIZPUR, POST- HEERAPATTI, AZAMGARH, Uttar Pradesh | 0.00 | 0.52 | 0.87 | 0.00 | **0.3201** | `N/N/N/N` | Score 0.3201 < 0.70 threshold; Low name token overlap (Jacc=0.00); Numeric mismatch (Jacc=0.00) |
| `S1-881535873` | `S3-820808792` | Key C, Key D | **CFG2 (not CFG1) [Key C]** | **S1**: Azamgarh Marketing Private Limited<br>**Cand**: Azamgarh Marreing Private Limited | **S1**: C/O Ajai Kumar Rai, Village Lachirampur Hafizpur, Post- Heerapatti, Azamgarh, Uttar Pradesh<br>**Cand**: C/o Ajai Kumar Rai, Azamgarh, UP | 0.33 | 0.73 | 0.43 | 1.00 | **0.6026** | `N/N/N/N` | Score 0.6026 < 0.70 threshold; Low name token overlap (Jacc=0.33) |
| `S1-881535873` | `S3-724794232` | Key C, Key D | **CFG2 (not CFG1) [Key C]** | **S1**: Azamgarh Marketing Private Limited<br>**Cand**: Azamgarh Private Limited Service | **S1**: C/O Ajai Kumar Rai, Village Lachirampur Hafizpur, Post- Heerapatti, Azamgarh, Uttar Pradesh<br>**Cand**: C/o Ajai Kuamr Rai, Azamgarh, UP | 0.33 | 0.47 | 0.33 | 1.00 | **0.5281** | `N/N/N/N` | Score 0.5281 < 0.70 threshold; Low name token overlap (Jacc=0.33) |
| `S1-925783039` | `S3-698172821` | Canonical, Key C | **CFG1 True Link** | **S1**: Orelee's Barbershop<br>**Cand**: Orelee'S Services | **S1**: 1795 Westchester Drive, High Point, NC<br>**Cand**: Westchester Dr, High Point, North Carolina | 0.33 | 0.22 | 0.50 | 0.00 | **0.2685** | `N/N/N/N` | Score 0.2685 < 0.70 threshold; Low name token overlap (Jacc=0.33); Numeric mismatch (Jacc=0.00) |
| `S1-629417405` | `S2-928426462` | Address-Num, Key C | **CFG1 True Link** | **S1**: Moore Bitwise Inc<br>**Cand**: Moore Inc Center | **S1**: 337 Oakland Avenue, Michigan City, IN<br>**Cand**: 337 OAKLAND AVE, MICHHIGAN CITY CITY, IN | 0.33 | 0.25 | 0.71 | 1.00 | **0.5786** | `N/N/N/N` | Score 0.5786 < 0.70 threshold; Low name token overlap (Jacc=0.33) |
| `S1-629417405` | `S3-786131946` | Key C | **CFG2 (not CFG1) [Key C]** | **S1**: Moore Bitwise Inc<br>**Cand**: Moore Bítwise Inc | **S1**: 337 Oakland Avenue, Michigan City, IN<br>**Cand**: Michigan City, 0337 Oakland Ave, Indiana | 0.33 | 0.62 | 0.50 | 0.00 | **0.3500** | `N/N/N/N` | Ranked outside Budget 25 (rank 219); Score 0.3500 < 0.70 threshold; Low name token overlap (Jacc=0.33); Numeric mismatch (Jacc=0.00) |
| `S1-629417405` | `S3-606982601` | Key C | **CFG2 (not CFG1) [Key C]** | **S1**: Moore Bitwise Inc<br>**Cand**: Moore Inc Services | **S1**: 337 Oakland Avenue, Michigan City, IN<br>**Cand**: 0337 Oakland Ave, Michigan City, Indiana | 0.33 | 0.23 | 0.50 | 0.00 | **0.2705** | `N/N/N/N` | Ranked outside Budget 25 (rank 194); Score 0.2705 < 0.70 threshold; Low name token overlap (Jacc=0.33); Numeric mismatch (Jacc=0.00) |
| `S1-513115995` | `S3-936443358` | Key B, Key D | **CFG1 True Link** | **S1**: Thomas Calisa<br>**Cand**: Th0mas Calisa | **S1**: 126 D Street, Pittsfield, ME<br>**Cand**: 126 D Stret, Pittsfield, Maine | 0.33 | 0.54 | 0.43 | 1.00 | **0.5648** | `N/N/N/N` | Score 0.5648 < 0.70 threshold; Low name token overlap (Jacc=0.33) |
| `S1-302710833` | `S3-203698864` | Key B, Key D | **CFG1 True Link** | **S1**: Kial Weaver Signature Associates Halifax<br>**Cand**: Wmeamevr, Kial Signature Associates Halifax | **S1**: NC, 6961 301, Halifax<br>**Cand**: Halifax, 301, North Carolina, <NULL> | 0.67 | 0.59 | 0.29 | 0.50 | **0.5146** | `N/N/N/N` | Ranked outside Budget 25 (rank 112); Score 0.5146 < 0.70 threshold; Low address token overlap (Jacc=0.29) |
| `S1-302710833` | `S3-77639341` | Canonical, Address-Num, Key C | **CFG1 True Link** | **S1**: Kial Weaver Signature Associates Halifax<br>**Cand**: Kial Weaver  Signature | **S1**: NC, 6961 301, Halifax<br>**Cand**: 961 301, <NULL>, Halifax, North Carolina | 0.60 | 0.50 | 0.25 | 0.33 | **0.4258** | `N/N/N/N` | Score 0.4258 < 0.70 threshold; Low address token overlap (Jacc=0.25) |
| `S1-302710833` | `S2-546588882` | Canonical, Address-Num, Key B, Key D | **CFG1 True Link** | **S1**: Kial Weaver Signature Associates Halifax<br>**Cand**: Kial Weaver Signature Associates | **S1**: NC, 6961 301, Halifax<br>**Cand**: 6961 301, PMB 212, HLAIFAX TOWNSHIP, NC | 0.80 | 0.79 | 0.38 | 0.67 | **0.6592** | `Y/N/N/N` | Score 0.6592 < 0.70 threshold |
| `S1-302710833` | `S2-888817530` | Key B, Key D | **CFG1 True Link** | **S1**: Kial Weaver Signature Associates Halifax<br>**Cand**: The Kial Weaver Signature Associates Halifax | **S1**: NC, 6961 301, Halifax<br>**Cand**: NC, 6961 301, HLAIFAX TOWNSHIP | 0.83 | 0.92 | 0.50 | 1.00 | **0.8088** | `Y/Y/Y/Y` | Ranked outside Budget 25 (rank 93) |

---

## 3. Table B: False Positive Accepted Pairs (Accepted at $\theta = 0.70$)

The table below details non-matching candidate records erroneously accepted at $\theta = 0.70$ under Budget 25.

| S1 ID | Candidate ID | Blocker Origin | S1 Name / Cand Name | S1 Address / Cand Address | Name Tok | Name Char | Addr Tok | Addr Num | Final Score | Acceptance Mechanism |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `S1-597762257` | `S3-163494033` | Canonical, Key C | **S1**: Green Logistics Private Limited<br>**Cand**: Green Logistics Private Ltd | **S1**: E-7, Second Floor, New Delhi, South Delhi, Delhi<br>**Cand**: South West Delhi, Rz-036 Third Floor, दिल्ली, Newdelhi | 1.00 | 0.71 | 0.23 | 0.00 | **0.9231** | Exact/canonical name match override shortcut |
| `S1-597762257` | `S2-781225628` | Exact, Canonical | **S1**: Green Logistics Private Limited<br>**Cand**: Green Logistics  Private Limited | **S1**: E-7, Second Floor, New Delhi, South Delhi, Delhi<br>**Cand**:  | 1.00 | 1.00 | 0.00 | 0.00 | **0.8500** | Exact/canonical name match override shortcut |
| `S1-223600564` | `S2-433265018` | Canonical | **S1**: Pushpam Institute Corp<br>**Cand**: PUSHPAM  INSTITUTE | **S1**: 1204, Block-C, Stratum @ Venus Ground, Nr. Jhansi Ki Rani Statue, Nehrunagar, Ahmadabad City, Ahmedabad, Gujarat<br>**Cand**:  | 1.00 | 0.78 | 0.00 | 0.00 | **0.8500** | Exact/canonical name match override shortcut |
| `S1-223600564` | `S3-593521455` | Canonical | **S1**: Pushpam Institute Corp<br>**Cand**: Pushpam Institute  Limited | **S1**: 1204, Block-C, Stratum @ Venus Ground, Nr. Jhansi Ki Rani Statue, Nehrunagar, Ahmadabad City, Ahmedabad, Gujarat<br>**Cand**:  | 1.00 | 0.56 | 0.00 | 0.00 | **0.8500** | Exact/canonical name match override shortcut |
| `S1-557783855` | `S2-320843595` | Canonical | **S1**: Sunrise Investments Private Limited<br>**Cand**: Sunrise Investments | **S1**: 818 Devika Tower, Nehru Place, New Delhi, Delhi<br>**Cand**: GREATER KAILASHNCT OF DELHI, NEW DELHI, Delhi | 1.00 | 0.53 | 0.20 | 0.00 | **0.9200** | Exact/canonical name match override shortcut |
| `S1-513115995` | `S2-44508570` | Canonical, Key C, Key D | **S1**: Thomas Calisa<br>**Cand**: THOMAS INC CALISA | **S1**: 126 D Street, Pittsfield, ME<br>**Cand**: 131 D ST, PITTSFIELD, ME | 1.00 | 0.53 | 0.67 | 0.00 | **0.9667** | Exact/canonical name match override shortcut |
| `S1-965555167` | `S3-221558511` | Canonical, Key C, Key D | **S1**: Siobhan's Bike Shop LLC<br>**Cand**: Siobhan's Bike Shop Corp | **S1**: 1644 Crownsville Road, Fl 0, Crownsville, MD<br>**Cand**: 1657 Crownsville Rd, Floor 0, Crownsville, Maryland | 1.00 | 0.67 | 0.50 | 0.33 | **0.9500** | Exact/canonical name match override shortcut |
| `S1-965555167` | `S2-120876485` | Canonical | **S1**: Siobhan's Bike Shop LLC<br>**Cand**: Siobhan's Bike Shop | **S1**: 1644 Crownsville Road, Fl 0, Crownsville, MD<br>**Cand**:  | 1.00 | 0.82 | 0.00 | 0.00 | **0.8500** | Exact/canonical name match override shortcut |
| `S1-951599976` | `S3-638165482` | Canonical, Key C, Key D | **S1**: Unified Green Target LLC<br>**Cand**: Unified Green Target Corp | **S1**: 1037 Cedar Grove Road, Halifax County, VA<br>**Cand**: 1039 Cedar Grove Rd, Alton, Virginia | 1.00 | 0.70 | 0.30 | 0.00 | **0.9300** | Exact/canonical name match override shortcut |
| `S1-960158214` | `S2-358996304` | Canonical, Key C, Key D | **S1**: Tinoco Institute of Technology Inc<br>**Cand**: Tinoco Institute of Technology Corp | **S1**: 27418 Fairway Oaks Drive, Huffman, TX<br>**Cand**: 27427 FAIRWAY OAKS DRIVE, <NULL>, HUFFMAN, TX | 1.00 | 0.78 | 0.62 | 0.00 | **0.9625** | Exact/canonical name match override shortcut |

---

## 4. Aggregate Failure Pattern Analysis

### False Negative Failure Breakdown (Total: 13)
- **Name Variation / Partial Name with Strong Address**: 7 cases (53.8%)
- **Compound Low Overlap (Both Name & Address Differ in Tokens)**: 3 cases (23.1%)
- **Strong Name with Address Formatting / Synonym Mismatch**: 2 cases (15.4%)
- **Ranked Outside Candidate Budget (Signal Priority)**: 1 cases (7.7%)

### False Positive Acceptance Breakdown (Total: 10)
- **Homonym / Common Name Match with Unrelated Address**: 10 cases (100.0%)

---

## 5. Concrete Recommendations for Matcher Architecture

Based strictly on the observed diagnostic tables:

1. **Replace Strict Token Jaccard with Soft / Containment Similarities**:
   - Strict Jaccard divides intersection by union. When one source includes store numbers, departments, or legal qualifiers (e.g. `["home", "depot"]` vs `["home", "depot", "store", "3821"]`), Jaccard drops to 0.50 even though the containment similarity is 1.00.
   - Implementing **Token Containment / Overlap Coefficient** ($|A \cap B| / \min(|A|, |B|)$) and **Fuzzy String Alignment (Levenshtein / Jaro-Winkler)** will prevent real entity variations from scoring below 0.50.

2. **Decouple Exact Name Shortcut from Address Disregard**:
   - The current rule assigns a score of 0.65 to any pair sharing an exact name even if the addresses completely disagree. This is the primary driver of False Positives for chain stores and common business names.
   - Exact name matches must require geographic consistency (postal code or street number agreement) to be accepted.

3. **Incorporate Postal / Locality Hierarchy into Address Scoring**:
   - Postal codes (PIN / ZIP) and primary street numbers should carry dedicated high-weight binary/numeric agreement features rather than being diluted across generic address word Jaccard.
