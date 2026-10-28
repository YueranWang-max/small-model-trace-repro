"""Offline cache geometry only; no runtime changes, model calls or performance scores."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
config_path = ROOT/'hybrid_candidate_qualification_20261004/config.json'
out = ROOT/'native_window_geometry_20261004.json'
assert not out.exists(), 'Keep evidence snapshots'
config = json.loads(config_path.read_text(encoding='utf-8'))['text_config']
full_layers = config['layer_types'].count('full_attention')
kv_heads = config['num_key_value_heads']
head_dim = config['head_dim']
assert (full_layers,kv_heads,head_dim) == (8,4,256)
f16_bytes_per_token = 2 * full_layers * kv_heads * head_dim * 2
windows = []
for context in [65536,131072,262144]:
    f16 = context * f16_bytes_per_token
    windows.append(dict(context=context, full_attention_f16_kv_bytes_geometry=f16,
                        f16_gib=f16/2**30,
                        q8_approx_gib_from_documented_half_ratio=f16/2/2**30))
case = dict(origin='Previously audited first-graph row506 input; diagnostic only, not a new independent case.',
            saved_trace_reader_input=144677, margin=64)
case['configured_old_reserve'] = 32768
case['old_required_positions'] = 144677+32768+64
case['native_window_margin_with_old_reserve'] = 262144-case['old_required_positions']
case['hypothetical_larger_reserve'] = 81920
case['required_positions_with_larger_reserve'] = 144677+81920+64
case['native_window_margin_with_larger_reserve'] = 262144-case['required_positions_with_larger_reserve']
result = dict(scope='Architecture-based position/cache arithmetic, not backend or quality qualification.',
              config_path=str(config_path),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
              full_attention_layers=full_layers,kv_heads=kv_heads,head_dim=head_dim,
              f16_kv_bytes_per_token_geometry=f16_bytes_per_token,windows=windows,
              archived_bundle_position_geometry=case, inference_calls=0,
              runtime_modified=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limits=['Full-attention KV geometry excludes recurrent state, weights, temporary workspaces, saved prompt caches, padding and allocator overhead.',
                      'Documented q8 approximate half-memory ratio omits quantization metadata/padding and does not prove actual backend allocation.',
                      'Nominal position fit is not VRAM fit, full GPU residency, successful long-input inference or preserved quality.',
                      'Increasing direct output allowance changes the cost bound; the previous fixed-cap single-prediction theorem is not universal.',
                      'Current frozen trial context, cache and output allowances are unchanged.'])
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
