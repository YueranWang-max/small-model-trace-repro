"""Finite-state schema removes free-text serialization as a chain bottleneck."""
import json,sys,time,urllib.request
from types import SimpleNamespace
import run_controlled_step_execution as base
ROOT=base.ROOT;OUT=ROOT/'logs/controlled_step_schema';SCHEMA={'type':'object','properties':{'state':{'type':'string','enum':[f'S{i:02d}' for i in range(8)]}},'required':['state'],'additionalProperties':False};original_prompt=base.prompt

def prompt(t,a,j,current):return original_prompt(t,a,j,current).replace('Return exactly Final Answer: <one state code S00 through S07>. No explanation.','Return a JSON object with exactly one field state, chosen from S00 through S07. No explanation.')

def native(model,system,user,**kwargs):
    payload={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'stream':False,'think':False,'format':SCHEMA,'options':{'temperature':0,'num_ctx':kwargs.get('num_ctx',32768),'num_predict':kwargs.get('num_predict',256),'top_p':.9}}
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));raw=None
    for attempt in range(3):
        try:
            with opener.open(req,timeout=900) as resp:raw=json.loads(resp.read().decode())
            break
        except Exception:
            if attempt==2:raise
            time.sleep(2)
    text=raw.get('message',{}).get('content','');raw['native_schema_content']=text;raw['client_attempts']=attempt+1;raw['request_payload_sha256']=base.prior.sha(json.dumps(payload,sort_keys=True).encode());value=None
    try:
        obj=json.loads(text)
        if isinstance(obj,dict) and set(obj)=={'state'} and obj['state'] in SCHEMA['properties']['state']['enum']:value=obj['state']
    except ValueError:pass
    raw['schema_valid']=value is not None
    return SimpleNamespace(content='Final Answer: '+value if value else '',thinking=raw.get('message',{}).get('thinking',''),raw=raw)

def freeze():
    base.OUT=OUT;base.freeze();p=OUT/'frozen.json';d=json.loads(p.read_text(encoding='utf-8'));d['adaptive_schema_diagnostic']=True;d['schema']=SCHEMA
    for file in [ROOT/'run_controlled_step_schema.py',ROOT/'protocol_controlled_step_schema_20261003.md']:d['hashes'][str(file.relative_to(ROOT))]=base.prior.sha(file.read_bytes())
    p.write_text(json.dumps(d,indent=2),encoding='utf-8')

def run():base.OUT=OUT;base.prompt=prompt;base.prior.ollama_chat=native;base.run()
def analyze():base.OUT=OUT;base.prompt=prompt;base.analyze()
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
