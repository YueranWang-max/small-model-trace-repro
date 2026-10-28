import audit_qasper_attention_reader as audit
from run_qasper_contract_fallback import OUT,ARMS
if __name__=='__main__':
    audit.OUT=OUT; audit.ARMS=ARMS; audit.main()
