# Majority baseline source check

Checked [Trace as State Appendix E, Figure 6](https://arxiv.org/html/2609.02702): Majority@5 is an element-wise threshold of three source prediction sets. The paper's main-table phrase alone was less explicit; the appendix resolves the rule. It is incorrect to describe the full paper as lacking a specified voting rule.

The existing local `audit_official_graph_five_trace.py`, line32, constructs a Counter over each valid source prediction set and retains nodes with count>=3. The archived local 57.50% F1 /25% EM baseline is **node-wise Majority@5**, not whole-set mode voting. A previous continuation summary mislabeled it as SetMode; original files and numerical results have not been changed.

The prospective new study retains NodeMajority with that threshold, invalid sources casting no votes, denominator fixed at five. Its programmatic empty aggregate is scored normally even if all five source outputs are invalid, consistent with the existing local implementation. All-invalid groups and the always-empty baseline will be disclosed. SetMode is an additional whole-set modal control, with earliest-source ties and abstention scored zero if there are no valid source answers. These are different algorithms.

The native trace serializer, warning, system instruction and answer-format suffix were also checked against Appendix C. Their texts and head50000-character clipping match the archived original definitions. Models, graph length strata, context allocation and output allowances still differ from the original frontier-model 256K experiment; the local study remains a bounded migration, not full replication of its complete experimental setting.
