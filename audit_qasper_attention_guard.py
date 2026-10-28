"""Audit frozen guard selections after selection, with no scoring feedback."""
import audit_qasper_attention_reader as audit
from run_qasper_attention_guard import OUT, ARMS

if __name__ == '__main__':
    audit.OUT = OUT
    audit.ARMS = ARMS
    audit.main()
