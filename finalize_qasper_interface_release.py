"""Audit transport recovery; retain frozen scores and publish stage provenance."""
import json
from pathlib import Path
from run_qasper_interface_replication import ROOT,OUT,MODELS,sha,save

def main():
    dest=ROOT/'logs/qasper_interface_transport_recovery';fr=json.loads((dest/'frozen.json').read_text(encoding='utf-8'));data=json.loads((dest/'results.json').read_text(encoding='utf-8'));s=json.loads((dest/'summary.json').read_text(encoding='utf-8'));primary=json.loads((OUT/'scored_results.json').read_text(encoding='utf-8'))
    assert fr['script_sha256']==sha((ROOT/'run_qasper_interface_transport_recovery.py').read_bytes()) and fr['protocol_sha256']==sha((ROOT/'protocol_qasper_interface_transport_recovery_20261003.md').read_bytes()) and fr['control_results_sha256']==sha((ROOT/'logs/qasper_interface_full_binding/results.json').read_bytes()) and data['frozen_sha256']==sha((dest/'frozen.json').read_bytes())
    assert set(data['arms'])=={'full_bind','wide_bind'}
    for a,r in data['arms'].items():
        assert r['request']==fr['requests'][a] and r['request_sha256']==sha(json.dumps(r['request'],sort_keys=True).encode()) and r['http_attempts']==1
        if 'error' not in r:assert json.loads(r['raw']['message']['content'])==r['parsed']
    row=next(r for r in primary['models'][s['model']]['records'] if r['id']==s['case_id']);original=row['arms']['wide_bind']['f1'];save(dest/'independent_audit.json',{'calls':2,'http_attempts':2,'requests_reconstructed':True,'audit_pass':True,'original_wide_bind_f1_recomputed':original,'repeated_wide_bind_f1':data['arms']['wide_bind']['f1'],'recovery_response_sha256':sha((dest/'results.json').read_bytes()),'frozen_primary_replaced':False})
    p=ROOT/'paper/QASPER_INTERFACE_REPLICATION_20261003.md';text=p.read_text(encoding='utf-8');text=text.replace('|qwen3:1.7b|27.73|150161|','|qwen3:1.7b|27.73|≥150161|')
    text+='\n## HTTP故障恢复：独立的两次诊断\n\n只因HTTP500选中1题，另行冻结后重放完全相同的full_bind和wide_bind请求，各1次；两次都成功，原失败仍保留0分，原门槛不变。失败原因只能确定为HTTP500，不能确定为长上下文、OOM或预算问题。恢复后全文同格式F1的事后敏感性均值为'+f"{100*s['full_bind_recovery_sensitivity_mean']:.2f}%"+'（原主记录27.73%），不是新的主成绩或独立确认。相同题wide_bind原分'+f'{original:.6f}'+'，重放分'+f"{data['arms']['wide_bind']['f1']:.6f}"+'。故障请求缺失的token回执无法由恢复请求重建，仍为未知成本。\n\n本阶段总计386正式/诊断HTTP调用＋2预热，其中336是新问题预设重复，48是自适应同格式全文控制，2是故障恢复诊断；不能混称386次独立确认实验。原336零失败，后48有1次HTTP500。R图继续显示未被替换的主记录。\n'
    p.write_text(text,encoding='utf-8');audit=json.loads((OUT/'release_audit.json').read_text(encoding='utf-8'));audit.update({'formal_calls':386,'prespecified_calls':336,'adaptive_full_control_calls':48,'transport_diagnostic_calls':2,'http_attempts':386,'warmups':2,'http500_errors_preserved':1,'unknown_token_calls':1,'primary_results_overwritten':False,'report_sha256':sha(p.read_bytes())});save(OUT/'release_audit.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
