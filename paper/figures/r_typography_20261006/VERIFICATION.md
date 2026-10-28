# R figure verification, 6 October 2026

The seven figures currently embedded by the manuscript were redrawn in R from ten byte-identical input tables. All seven standalone redraw SVGs match their primary SVGs byte for byte. Redraw used a fresh R process on the same host with the documented fonts and libraries; this is not external replication. No model-generated imagery or Python graphics device was used. This is a measured figure-readability and reproducibility improvement, not evidence that experimental accuracy increased.

| Figure | Previous minimum at final width (pt) | Revised minimum (pt) | PDF font |
|---|---:|---:|---|
| interrupted_indexed_figure | 4.33 | 5.25 | TrueType / Arial |
| interrupted_study_figure | 3.90 | 5.25 | TrueType / Arial |
| official_graph_confirmatory_20261006 | 3.55 | 5.25 | TrueType / Arial |
| official_graph_identity_recode | 3.90 | 5.25 | TrueType / Arial |
| official_graph_reasoning_mode | 5.42 | 5.26 | TrueType / Arial |
| partial_state_display | 4.44 | 5.25 | TrueType / Arial |
| source_qualification_figure | 4.28 | 5.26 | TrueType / Arial |

Previous refers to the separately archived 5–6 October R release, not an undocumented initial image. Every revised PDF has embedded TrueType text, each SVG retains editable text nodes, and final-width body text falls within approximately 5.25–6.99 pt. R PNG previews were visually reviewed; clipped zero labels, cropped axis labels and coverage-footer overlap were corrected. There is no text bounding box outside the PDF page. Quantitative data, planned denominators and uncertainty definitions are unchanged.

Machine-readable evidence: font_metadata_qa.json, before_after_font_evidence.json, input_identity_qa.json and portable_redraw_verification.json in this release folder. The portable R source and inputs are in portable/. The pending paired-model plot is excluded until its completed evidence audit passes.
