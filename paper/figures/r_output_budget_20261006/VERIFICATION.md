# Output-budget figure verification

Rendered only after the 60-call diagnostic and 240-call comparator passed full independent audits. R produces the PDF, editable SVG and 600-dpi PNG. Python inspects text, fonts, file identity and XML only. The two panels show paired graph-mean direct F1 and preparation-inclusive mean logical tokens. No interval or population equivalence is inferred from a score ceiling; this is a sequential known-question diagnostic, not equal-compute causality.

Six copied inputs (four CSV tables and two complete audit summaries) are byte-identical. A fresh R process using the portable copied bundle produces a byte-identical SVG. R PNG inspected and multiline-label overlap corrected. At a final width of 6.5 inches, text is 5.584 to 6.980 points, Arial fonts are embedded, all 43 SVG text nodes remain editable, and no PDF text bounding box falls outside the page. Same-host redraw is not external inference replication.

See input_identity_qa.json, font_metadata_qa.json and portable_redraw_verification.json. Plotting source: portable/output_budget_20261006.R; entry point: portable/render_all.R.
