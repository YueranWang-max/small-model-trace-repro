"""Read-only structural/runtime diagnostic. No model calls or answer scoring."""
import datetime
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    destination = ROOT/'owned_cache_diagnostic_20261004.json'
    assert not destination.exists()
    source = ROOT/'owned_ollama_11435_stderr_20261004.log'
    data = source.read_bytes()
    lines = data.decode('utf-8',errors='replace').splitlines()
    # Never export whole environment/config lines, which can contain proxy credentials.
    chosen = [(i+1,line) for i,line in enumerate(lines) if 'server config' not in line and any(
        text in line for text in ['prompt cache','cache state:','kv cache removal','cached n_tokens'])]
    config = next((line for line in lines if 'server config' in line),'')

    def field(name):
        match = re.search(re.escape(name)+r':([^ ]*)',config)
        return match[1] if match else None

    frozen = ROOT/'partial_state_study_20261004/frozen.json'
    result = dict(status='live_readonly_cache_diagnostic_not_performance_analysis',
                  captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_path=str(source),source_bytes_as_read_sha256=hashlib.sha256(data).hexdigest(),
                  source_bytes_read=len(data),selected_cache_lines=chosen[-40:],
                  kv_cache_environment_value=field('OLLAMA_KV_CACHE_TYPE'),
                  flash_attention_environment_value=field('OLLAMA_FLASH_ATTENTION'),
                  frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  inference_calls=0,answer_scores_read=False,
                  limitations=['Live append-only log read is a byte snapshot, not an atomic future full-log certificate.',
                               'Logged cache states/events do not prove causal latency savings or exact per-call cached work.',
                               'No active protocol, environment or serving process changed.'])
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],source_bytes_read=len(data),
                         cache_lines_saved=len(result['selected_cache_lines']),
                         kv_cache_environment_value=result['kv_cache_environment_value'],
                         flash_attention_environment_value=result['flash_attention_environment_value']),indent=2))


if __name__ == '__main__':
    main()
