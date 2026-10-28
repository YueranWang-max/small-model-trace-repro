import pathlib, urllib.request, hashlib, json
ROOT=pathlib.Path(__file__).resolve().parent
items=[('graphwalks_128k_and_shorter.parquet','https://huggingface.co/datasets/openai/graphwalks/resolve/be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7/graphwalks_128k_and_shorter.parquet','54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'),('tokenizer.json','https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json','aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4')]
records=[]
for name,url,expected in items:
    p=ROOT/name
    if not p.exists():
        with urllib.request.urlopen(url,timeout=180) as response: data=response.read()
        assert hashlib.sha256(data).hexdigest()==expected, name
        p.write_bytes(data)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected, name
    records.append(dict(path=str(p),url=url,sha256=actual,bytes=p.stat().st_size))
    print(name,p.stat().st_size,actual,flush=True)
(ROOT/'bridge_data_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
