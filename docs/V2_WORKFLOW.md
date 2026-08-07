# Cell Vision V2 workflow

V2 preserves the existing well-level growth decision and replaces the object layer.

1. The recall-first V1 candidate generator supplies seeds across the valid well interior and the searchable wall buffer.
2. `SeededInstanceUNet` receives raw pixels, one seed heatmap, and a wall prior; it predicts the complete selected-object mask and the physical-wall mask.
3. Mask IoU and containment merge repeated proposals. A doublet/cluster mask owns overlapping single proposals, preventing an internal single-cell proposal from affecting the single-source decision.
4. Compact high-confidence masks can rescue V1 wall/invalid candidates, while elongated high-wall-overlap masks are rejected. This is what preserves real cells touching or overlapping the wall without admitting wall arcs.
5. The learned temporal evidence model consumes aligned T0/T1/T2 raw patches, masks, displacement, area change, and multiplicity. It outputs same-object, live-cell, static-debris, dead-cell and insufficient probabilities. Low-confidence or unsupported classes abstain as `insufficient`.
6. The quick-review UI initially downloads 1400 px cached JPEGs, upgrades to 4096 px only above 2.5x zoom, and draws the predicted instance contour. Statistics, well lists and details reuse one database-aware frame cache.
7. Fixed reports include target miss/recall, duplicate rate, wall false positives, multiplicity confusion, near-wall recall, timepoint recall, cross-plate differences, correction burden and review duration. Well-level sensitivity/specificity/PPV remain unavailable until a frozen human well-level reference is supplied.

Training sources are QL11111 and QL2202 reviewed data. A12-22 is explicitly excluded from weight updates and remains external validation.
