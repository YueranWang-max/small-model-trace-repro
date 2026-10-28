# Reproducing the tested local packages

The published variant name is insufficient to identify the tested model. Preserve the exact manifest digest, template, parameters, tokenizer and runtime in addition to the GGUF weight digest. `local_component_checks.json` verifies all local components against the archived manifests. Model weights are not in the evidence candidate.

The Qwen3.5 public tag changed after the local package was installed. Pulling the current `qwen3.5:4b` tag does not recreate the tested package. The archived manifest and component records identify the old package; its old public manifest endpoint returned 404 during the recorded check, although the content-addressed weight endpoint was available. Do not represent a newly created tag from the same weight alone as manifest-identical: template, configuration and parameter layers can differ.

Original Modelfiles preserve their original absolute FROM paths as provenance. These paths must be adapted on another host. A locally adapted Modelfile is a new artifact, and its model digest must be checked rather than assumed equal to the original. No inference re-execution on another host has been performed. Offline relocated audits verify the saved records, not the authenticity of historic execution or bitwise reproducibility of future GPU sampling.

The three local manifests tested were:

- Qwen3-1.7B: 8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7
- Qwen3-4B: 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7
- Qwen3.5-4B: 2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd

Every new experiment checks the resident package and configured context against its frozen protocol. Failed checks are preserved rather than relabelled as valid observations.
