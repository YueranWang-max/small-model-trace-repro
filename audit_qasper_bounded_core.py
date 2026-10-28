import audit_qasper_semantic_slots as audit
from run_qasper_bounded_core import OUT

if __name__=='__main__':
    audit.OUT=OUT
    audit.main()
