# Figure evidence and export contract, 6 October 2026

Backend: R only. Data extraction and file/metadata inspection may use Python; no Python graphics device or raster redraw is used. No generated imagery is used.

## Scope and evidence logic

All existing empirical figures retain the same audited CSV inputs. This revision changes typography and export, not observations, scoring, axes data or statistical estimands. Archetype: quantitative grids with restrained blue, orange and neutral encodings, readable labels and a white background.

| Figure | Conclusion and panel roles | Statistical risk controlled |
|---|---|---|
| Native source qualification | Graph-wise source performance; F1 and exact match separately | Eight independent graphs, two within-graph repeats; no inferential bars |
| Interrupted native trial | Incomplete quality, first-case configured overflow and actual slot coverage | Identification ranges are not confidence intervals; missing outcomes remain missing |
| Early-state adaptation | Incomplete outcome bounds, unequal coverage and incomplete cost lower bounds | Does not demonstrate a gain; post hoc parsing remains separate |
| Indexed source | Representation-specific quality coverage and known token usage | Preserves two missing responses and incomplete-cost arrows |
| Identity recoding | Paired graph variation, descriptive intervals and token composition | Eight graph units; exploratory contrasts remain labelled |
| Thinking mode | Task/length heterogeneity and quality-versus-cost trade-off | Route is derived from saved subsets; not a separately run arm |
| Prospective 1.7B | Per-graph heterogeneity, placement intervals and full preparation cost | Twelve graph units, five repeats; primary and secondary contrasts distinct |
| Output-budget diagnostic | Paired direct graph F1 and logical deployment tokens | Same known questions; no byte-identical continuation, equal-compute or population-equivalence claim |
| Paired 1.7B / 4B (both complete audits) | F1, exact match, within-model placement contrasts and preparation-inclusive tokens | Same twelve questions, not 24 independent graph units; do not claim size-only causality |

No pending experiment is plotted as completed. The paired plot was rendered after both complete 240-call audits passed. The completed output-budget diagnostic has a ninth R figure with paired direct F1 and preparation-inclusive logical cost. Its six copied inputs and standalone SVG redraw are verified. The context diagnostic remains pending and is planned for a case-specific table, not an inferential plot.

## Final size and export

The arXiv article uses about 6.5 inches of full text width; the source qualification figure uses 0.92 of that width. This is an arXiv layout with Nature-informed readability, not a certified Nature production submission. Derive the R device point size from the final manuscript scale so ordinary labels remain at least 5 pt, with the usual text around 5–7 pt. Lowercase panel identification and concise headings improve consistency.

Export editable SVG with svglite, vector PDF with R cairo_pdf and Arial font embedding, and a 600-dpi PNG with the same R drawing function. The paper embeds vector PDFs. Check the R-generated previews, selectable text, fonts and final-scale sizes. Preserve the old dated releases and create a separately versioned source/input/export package with SHA-256 hashes. Do not use synthetic or imputed data.

Official reference: [Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/), checked 6 October 2026. Its production dimensions and 5–7 pt typography are reference points; arXiv server processing and journal acceptance are separate checks.
