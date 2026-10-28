"""Post-selection audit of the separate schema-constrained cohort."""
import audit_qasper_semantic_slots as audit
from run_qasper_schema_core import OUT

if __name__=='__main__':
    audit.OUT=OUT
    audit.main()
