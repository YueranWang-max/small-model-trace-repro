# Early native state study: interrupted evidence

## Audited status

The previously registered 16-graph, two-repeat, seven-reader-arm study was interrupted. The exact termination time and cause are unknown. A read-only process inventory on 2026-10-05 confirmed the runner was absent. Its native terminal handle was unavailable. Record 150 contains a prepared request but no returned response; transmission and consumed usage cannot be determined from that record alone. No missing request was replayed.

The recovery completion record explicitly identifies external observation rather than runner completion. Actual complete workflow duration is unavailable, not reconstructed as a successful run.

The frozen independent audit passed: 150 saved records, 149 qualified responses including calibration and source calls, 129 known reader results out of 224 planned, and one attempted request with unknown usage. Known research usage is at least 3,621,241 tokens. This is a lower bound, not total completed-study cost.

## What the evidence supports

All planned predictions remain in the denominator. Unknown results have [0,1] outcome bounds. The primary prefix arm has F1 bounds [0.34375, 0.78125], versus direct [0.195855, 0.633355]. These are identification bounds from missing outcomes, not confidence intervals. Their overlap prevents a defensible all-plan superiority conclusion. No completed-study efficiency comparison or confirmatory gate pass is established.

Observed prefix normal-valid outputs are 11/18 known; direct 7/18 known. These descriptive counts do not establish an effect: completion status, accuracy and interrupted coverage differ, and the sample is incomplete. The prescribed primary reliability threshold of at least 31/32 is already unattainable because seven observed prefix outcomes were not normal-valid. Therefore completing the remaining slots alone could not make this configuration pass its original gate. This does not invalidate every partial-state method.

Frozen audit and analysis were executed without modification. The display export and base-R renderer succeeded. The resulting PNG was visually inspected: labels, bounds, cost arrows and disclaimers are legible, with no observed clipping. All displayed costs are lower bounds divided by 32 planned predictions per arm; state deployments charge their entire source call. Do not interpret lower-cost bars as completed deployment savings.

## Artifacts

- partial_state_study_20261004/completion.json: externally observed interruption provenance.
- partial_state_study_20261004/independent_audit.json and analysis.json: frozen audit and analysis outputs.
- partial_state_study_20261004/display_release.json: input/output hashes.
- partial_state_study_20261004/partial_state_display.pdf and .png: R-generated visual evidence.

No manuscript source was changed in this step. No new GPU experiment was started.
