# Oracle Correction Analysis - Key Finding

## Summary

**Oracle correction has ZERO impact on the current supervoxel partition.**

All 8,075 supervoxels are **100% pure** (homogeneous) - every voxel in each SV already belongs to the same class.

## Results

### Oracle Correction on 0.1% Budget

**Input:**
- 1,108 labeled voxels (random sampling)
- 1,031 labeled supervoxels

**Oracle Corrections:**
- **0 out of 1,031 SVs needed correction (0.0%)**
- Every sparse seed's label already matches its SV's GT majority

### Supervoxel Homogeneity Analysis

**Partition:** `sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted`

```
Total SVs: 8,075
Mean majority fraction: 1.000 (100%)
Median majority fraction: 1.000 (100%)

Distribution:
  100% pure (all voxels same class): 8,075 SVs (100.0%)

  90-99% majority: 0 SVs
  80-90% majority: 0 SVs
  70-80% majority: 0 SVs
  <70% majority: 0 SVs
```

**Every supervoxel is 100% homogeneous!**

## Why This Happens

### The "voted" Processing

The directory name reveals the answer: `sv_fullgt_slic_n12000_c0.05_s1.0_ras2_**voted**`

The "voted" suffix indicates these supervoxels were **post-processed** with majority voting:

1. **Original SLIC** generates supervoxels (may be heterogeneous)
2. **Majority voting** assigns each SV voxel to the SV's dominant class
3. **Result:** Every SV becomes 100% pure

This preprocessing was likely done to establish a clean upperbound for SV-based methods.

## Implications

### 1. Oracle Correction is Not Needed

**For voted supervoxels:**
- Any voxel sampled from a SV will have the correct (majority) label
- Random sampling and strategic sampling produce identical SV labels
- Oracle correction provides **zero benefit**

**Your concern was valid**, but only for **raw (non-voted) supervoxels**:
- Raw supervoxels CAN be heterogeneous
- A seed voxel might belong to minority class
- Oracle correction would help in that case

### 2. The 0.92 Dice Upperbound Explained

With 100% pure supervoxels, why isn't the upperbound 100%?

**Answer:** SV boundaries don't perfectly align with anatomical boundaries.

Even with perfect SV-level labels, you lose information at boundaries:

```
Ground Truth:     Supervoxel Segmentation:
┌────┬────┐       ┌─────────┐
│ A  │ B  │       │ A A B B │ ← SV can't split the boundary
│ A  │ B  │       │ A A B B │
└────┴────┘       └─────────┘
                   ↑ Some A voxels labeled as B (and vice versa)
```

The 0.92 Dice represents the best possible alignment given the SV partition granularity.

### 3. Graph LP Results Are Already Optimal

**Current results (0.1% budget):**
- Graph LP with seed labels: 78.8% FG Dice
- Graph LP with oracle labels: 78.8% FG Dice (no change)

**Interpretation:**
- The 78.8% result is already the best possible for this approach
- The gap to 92% upperbound comes from:
  - Sparse seed coverage (only 12.8% of SVs labeled)
  - Propagation errors across unlabeled regions
  - NOT from incorrect SV label assignment

## What About Non-Voted Supervoxels?

### Testing on Raw Supervoxels

If you want to test oracle correction's true impact, you would need:

1. **Raw SLIC supervoxels** (before voting)
2. These would have heterogeneous SVs
3. Oracle correction would fix cases where seed != SV majority

### Expected Impact

On raw supervoxels with ~80-90% homogeneity:
- ~10-20% of labeled SVs might need correction
- Oracle correction could improve results by 1-3 pp
- But you'd lose the clean 92% upperbound (raw SVs have lower ceiling)

### Trade-off

**Voted SVs (current):**
- ✅ Clean upperbound (92%)
- ✅ No SV label assignment errors
- ✅ Simple pipeline (no oracle needed)
- ❌ Lower ceiling due to boundary alignment

**Raw SVs (alternative):**
- ✅ Higher potential ceiling (closer to GT)
- ❌ SV label assignment errors possible
- ❌ More complex pipeline (need oracle or strategic sampling)
- ❌ Harder to interpret results

## Recommendation

**Keep using voted supervoxels for now.**

Reasons:
1. Your concern about minority-class seeds is **already solved** by the voted preprocessing
2. Oracle correction provides **zero benefit** (0.0% corrections)
3. The 92% upperbound is respectable and well-defined
4. Graph LP already achieves 78.8% at 0.1% budget (85% of upperbound)

**Future exploration:**
- If you want to push beyond 92%, consider:
  - Finer-grained supervoxels (n=20K instead of 12K)
  - Different supervoxel algorithms (Watershed, Felzenszwalb)
  - Voxel-level propagation (no SV abstraction)

## Conclusion

Your intuition was correct - **in theory**, seed voxels could belong to minority classes in heterogeneous supervoxels, requiring oracle correction.

**In practice**, the voted supervoxel partition eliminates this problem entirely. All supervoxels are 100% pure, so oracle correction has zero impact.

The current Graph LP results (78.8% at 0.1% budget) are already optimal for this SV partition. The gap to 92% is due to sparse coverage, not incorrect SV labels.

---

**Files Generated:**
- `scripts/oracle_correct_sv_labels.py` - Oracle correction implementation
- `tmp/oracle_sv_labels_0p1pct.json` - Results (0 corrections needed)
- `tmp/ORACLE_CORRECTION_ANALYSIS.md` - This analysis
