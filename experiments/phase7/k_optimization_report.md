# Phase 7: Priority 2 - Top-K Optimization

## 1. Experimental Methodology
We performed a highly efficient isolation of the K truncation effect. The inverted index was loaded once and all 220,683 validation S1 entities were queried using an unbounded budget (`K=100000`). The full ranked list of candidates was maintained in memory, and the results were evaluated by slicing the lists at K=100, 150, 200, 300, 500, and UNLIMITED. This isolated the effect of the K parameter entirely without recompiling the candidate pool.

## 2. Experimental Results (Validation Split, N=220,683)
*Total Possible GT Links: 763,592*

| K | Blocking Recall | Total Cands | Avg Cands/S1 | P95 | Incremental GT Recovered (vs 100) | Incremental Cands (vs 100) |
|---|---|---|---|---|---|---|
| **100** | 74.29% | 10.0M | 45.4 | 100 | - | - |
| **150** | 75.99% | 12.8M | 58.3 | 150 | +12,969 | +2.8M (+28%) |
| **200** | 77.04% | 15.1M | 68.6 | 200 | +20,991 | +5.1M (+51%) |
| **300** | 78.01% | 18.6M | 84.5 | 300 | +28,390 | +8.6M (+86%) |
| **500** | 78.98% | 23.8M | 107.8 | 500 | +35,791 | +13.8M (+138%) |
| **UNLIMITED**| 80.28% | 34.7M | 157.3 | 755 | +45,689 | +24.7M (+247%) |

## 3. Top-K Rank-Loss Distribution
This is the distribution of the ranks of GT links in the full, un-truncated candidate lists:

- **1-100:** 567,291
- **101-150:** 12,969 *(High density)*
- **151-200:** 8,022 *(Moderate density)*
- **201-300:** 7,399 *(Low density)*
- **301-500:** 7,401 *(Very low density)*
- **>500:** 9,898 *(Extremely long tail)*
- **Not Generated:** 150,612 *(Complete blocker failure)*

## 4. Observations
The rank distribution reveals a distinct saturation point around `K=200`. Between ranks 100 and 200, we recover a dense cluster of ~21,000 GT links. However, past rank 200, the true matches become extremely sparse. For example, expanding K from 300 to 500 (adding 200 more slots) recovers only 7,401 links, which means the vast majority of those extra 5.2 million candidates are useless look-alikes.

## 5. Final Recommendation

**RECOMMENDATION:**
K = 200 (or investigate Adaptive Top-K)

**Blocking recall:**
77.04%

**Candidates:**
15,143,621

**Additional recall vs K=100:**
+2.75% absolute (20,991 links)

**Candidate increase vs K=100:**
+5.1 million (+51%)

**GT links still lost:**
175,310

**Reason:**
`K=200` is the "sweet spot" on the recall-volume curve. It captures nearly 46% of the total available gain from removing the budget while only increasing candidates by a modest 50%. Past `K=200`, the density of true matches collapses: moving to `K=500` would add another 8.7 million candidates just to recover 14,800 links. If we wish to recover the remaining 25,000 links beyond rank 200, we should investigate an adaptive Top-K strategy (e.g. only expanding K if the candidate score stays high) rather than blindly forcing a massive fixed K limit.

*Note: 150,612 GT links are still completely absent from the pre-Top-K candidate set, representing hard limits in the current index keys.*
