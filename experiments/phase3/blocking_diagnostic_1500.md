# Blocking Diagnostic (1500-S1 Dev Set)

## 1. CFG1 Missed GT Links
**Total missed GT links**: 1167

- Recovered by Key C (CFG2): 204
- Recovered by Key D (CFG3 \ CFG2): 87
- Still Missed (NO recovery by CFG3): 876

- Missed links lacking shared numeric address tokens: 440 (37.7%)
- Average Name Similarity (SequenceMatcher): 0.494
- Missed links with <0.5 Name Similarity: 522

### Analysis of Causes for Missed Links
Based on the data above, the missed links likely fall into these categories:
- **Low Name Similarity + No Address Numerics**: The blocker requires a shared word or number. If the name is completely different (e.g., transliteration mismatch, or brand vs legal name) AND there is no shared numeric token in the address, CFG1 will completely miss it. (Key C and Key D also cannot recover these).
- **Second-Token Divergence**: S1 and GT share the first word, but differ on the second word. CFG1 misses this if `idx_two_tok` is the only active key. CFG2 (Key C - first_tok + salient_alpha) can recover some of these.
- **Address Format Mismatch**: When names don't exactly match and addresses don't have overlapping parsed numerics.

### Sample Missed Links (Top 15 lowest similarity)
| S1 Name | GT Name | S1 Addr | GT Addr | Sim | Num? | Rec? | CFG1_Cands |
|---|---|---|---|---|---|---|---|
| Hitech Infrastructure | हाईटेक इंफ्रास्ट्रक्चर | Bluridge Township Ha-4, F No G1 (A1), Haveli, Pune, Maharashtra | F No G1 (A1), Haveli, MH, Pune, Door No 767 Bluridge Township Ha-4 | 0.05 | N | None | 64 |
| Hitech Infrastructure | हाईटेक इंफ्रास्ट्रक्चर | Bluridge Township Ha-4, F No G1 (A1), Haveli, Pune, Maharashtra | BLOCK D-0836 BLURIDGE TOWNSHIP HA-4, HAVELI, PUNE, Maharashtra | 0.05 | N | None | 64 |
| Premier Technologies | प्रीमियर टेक्नोलॉजीज | G-9/10, Ground Floor, Shree Industrial Estate, V P Road, Dombivali, Thane, Maharashtra | G-9/10/8, GROUND FLOOR, SHREE INDUSTRIAL ESTATE, V P ROAD, DOMBIVALI, THANE, Maharashtra | 0.05 | Y | None | 268 |
| Premier Technologies | प्रीमियर टेक्नोलॉजीज | G-9/10, Ground Floor, Shree Industrial Estate, V P Road, Dombivali, Thane, Maharashtra | G-9/10, Dombivali, Thane, MH | 0.05 | Y | None | 268 |
| Lds Engineering Private Limited | Zephhalo | Abjibapa Lake View, Nirant Chokdi, Daskroi, Ahmedabad, A-404, Gujarat | A-404, Ahmedabad, Daskroi, GJ | 0.05 | Y | None | 49 |
| Classic Consultancy | क्लासिक कंसल्टेंसी | Office No.54 A/B, 5Th Floor, Jolly Maker Chambers No.Ii, 225, Vinay K.Shah Marg, Nariman P, Oint, Mumbai, Mumbai City, Maharashtra | Office No.54 A/b, 5Th Floor, Jolly Maker Chambers No.ii, 225, Vinay K.shah Marg, Nariman P, Oint, Mumbai, Mumbai City, MH | 0.05 | Y | None | 124 |
| Classic Consultancy | क्लासिक कंसल्टेंसी | Office No.54 A/B, 5Th Floor, Jolly Maker Chambers No.Ii, 225, Vinay K.Shah Marg, Nariman P, Oint, Mumbai, Mumbai City, Maharashtra | Maharashtra, OFFICE NO.52 A/B, MUMBAI | 0.05 | Y | None | 124 |
| Dream Investments | ड्रीम इन्वेस्टमेंट्स | Plot No. 305 First Floor Industrial Area Phase 1, Panchkula, Haryana | 305 First Floor Indutsrial Area Phase 1, Panchkula, हरियाणा | 0.05 | Y | None | 90 |
| Dream Investments | ड्रीम इन्वेस्टमेंट्स | Plot No. 305 First Floor Industrial Area Phase 1, Panchkula, Haryana | #305. FIRST FLOOR INDUSTRIAL AREA PHASE 1, PANCHKULA, Haryana | 0.05 | Y | None | 90 |
| Lotus Engineering | லோட்டஸ் இன்ஜினியரிங் | No.9, 1St Floor, Duraisamy Street Nungambakkam, Chennai, Tamil Nadu | H.NO 9, CHENNAI, Tamil Nadu | 0.06 | N | None | 66 |
| Lotus Engineering | லோட்டஸ் இன்ஜினியரிங் | No.9, 1St Floor, Duraisamy Street Nungambakkam, Chennai, Tamil Nadu | No.9, <NULL>, Chennai, TN | 0.06 | N | None | 66 |
| Marylin's Clear Systems LLC | Deltalum | 111 Clinton Avenue, Dunn, NC | 111 Clinton Ave, Dunn, North Carolina | 0.06 | Y | None | 99 |
| Surya Consultants | ಸೂರ್ಯ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ | No 1202, 12Th Floor, Birla Apple Spire, Nayandahalli, Bangalore South, Bangalore, Karnataka | NO 1202, 12TH FLOOR, BIRLA APPLE SPIRE, NAYANDAHALLI, BANGALORE URBAN, <NULL>, Karnataka | 0.06 | Y | None | 129 |
| Real Construction | रियल कंस्ट्रक्शन | G-2, 44/45, Sdc Retreat Shrirampura Colony, Shivaji Nagar, Civil, Lines, Jaipur, Rajasthan | No. 551 G-2, Jaipur, RJ | 0.06 | Y | None | 1071 |
| Real Construction | रियल कंस्ट्रक्शन | G-2, 44/45, Sdc Retreat Shrirampura Colony, Shivaji Nagar, Civil, Lines, Jaipur, Rajasthan | G-C-2, 44/45, SDC RETREAT SHRIRAMPURA COLONY, SHIVAJI NAGAR, CIVIL, LINES, JAIPUR, Rajasthan | 0.06 | Y | None | 1071 |

## 2. CFG1 Candidate Explosions (>10k candidates)
**Total entities with >10k candidates**: 31

### Cross-Tabulation: Explosions vs Missed Links
- Number of Missed GT Links that came from 'explosion' S1 entities: 13
- **Conclusion**: The missed links and the explosions are largely distinct problems. High candidate counts come from very generic entities, while missed links are mostly specific businesses with severe name/address discrepancies or lacking shared tokens.

### Top Explosions (by total candidates)
| Country | S1 Name | GT Links | Total Cands | Exact Index Cands | Canon Index Cands | TwoTok Index Cands |
|---|---|---|---|---|---|---|
| US | Ear Nose & Throat Specialists | 5 | 18693 | 138 | 474 | 18514 |
| US | Ear Nose & Throat Highland Physicians Inc | 8 | 18630 | 2 | 11 | 18514 |
| US | Ear Nose & Throat Partners Inc | 3 | 18601 | 60 | 366 | 18514 |
| US | Ear Nose & Throat Care Associates Inc. | 4 | 18601 | 56 | 346 | 18514 |
| US | Ear Nose & Throat Global Care Associates LLC | 3 | 18541 | 1 | 4 | 18514 |
| US | Ear Nose & Throat Center Group | 2 | 18525 | 15 | 27 | 18514 |
| US | Foot & Ankle Federal Medicine LLC | 3 | 18025 | 3 | 12 | 17776 |
| US | Physical Therapy Care Associates Inc | 2 | 17731 | 59 | 433 | 17628 |
| US | Physical Therapy Specialists LLC | 4 | 17655 | 91 | 509 | 17628 |
| US | Primary Care Complete Health LLC | 3 | 17652 | 4 | 14 | 17562 |
| US | Primary Care Health PLLC | 4 | 17626 | 4 | 542 | 17562 |
| US | Primary Care Gulf Clinic LLC | 3 | 17625 | 0 | 2 | 17562 |
| US | Pediatric Dentistry Specialists LLC | 4 | 17385 | 105 | 466 | 17351 |
| US | Pediatric Dental Care Associates | 3 | 17380 | 125 | 494 | 17332 |
| US | Pediatric Dentistry Signature Care Associates LLC | 6 | 17374 | 1 | 9 | 17351 |
| US | Pediatric Dental Clinic Inc | 3 | 17367 | 75 | 453 | 17332 |
| US | Pediatric Dental Pioneer Specialists Inc. | 2 | 17357 | 0 | 3 | 17332 |
| India | New Delhi Studios Private Limited | 5 | 17016 | 1 | 10 | 16492 |
| US | Behavioral Health Specialists | 3 | 16936 | 158 | 353 | 16886 |
| US | Behavioral Health Medicine LLC | 5 | 16931 | 76 | 333 | 16886 |
| US | Urgent Care Center LLC | 3 | 16624 | 78 | 562 | 16563 |
| US | Urgent Care Medicine LLC | 4 | 16619 | 96 | 532 | 16563 |
| US | Internal Medicine Specialists LLC | 5 | 16619 | 87 | 348 | 16555 |
| US | Urgent Care Clinic Group | 4 | 16592 | 24 | 34 | 16563 |
| US | Internal Medicine Reliable Physicians | 3 | 16566 | 3 | 10 | 16555 |
| US | Urgent Care Prairie Physicians LLC | 4 | 16564 | 4 | 10 | 16563 |
| US | Internal Medicine Better Care Corp | 4 | 16555 | 0 | 3 | 16555 |
| US | Womens Health Direct Group | 6 | 16505 | 2 | 8 | 16499 |
| US | Tri-State Fund | 2 | 13865 | 33 | 68 | 13862 |
| US | Tri-State Rapid Gaming Inc | 3 | 13862 | 2 | 2 | 13862 |
