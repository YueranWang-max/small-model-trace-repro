"""Read backend flags without exporting environment lines or model outputs."""
import datetime
import hashlib
import json
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent
out = ROOT/'owned_backend_flags_20261004.json'
assert not out.exists()
data = (ROOT/'owned_ollama_11435_stderr_20261004.log').read_bytes()
lines = [line for line in data.decode('utf-8',errors='replace').splitlines()
         if 'server config' not in line and any(word in line.lower() for word in
                                               ['flash_attn','flash attention','type_k','type_v','kv buffer','kv cache size'])]
result = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_bytes_as_read_sha256=hashlib.sha256(data).hexdigest(),
              diagnostic_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              selected_backend_lines=lines,inference_calls=0,answer_scores_read=False,
              scope='Startup backend logs; environment boolean is not the effective Flash Attention setting.')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
