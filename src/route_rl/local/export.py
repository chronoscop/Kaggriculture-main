"""Bundle the actual baseline, local contract executor and event policy."""
import argparse,hashlib,io,json,math,os,tarfile,tempfile
from pathlib import Path
from ..paths import BASELINE
from .settings import SCHEMA
from .archive import MAX_ARCHIVE,MAX_DISK
from .encoding import CONTEXT_SIZE,CANDIDATE_SIZE
from .policy import Policy,greedy
from .runtime import NativePolicy
from .training import read_checkpoint

ENTRY = '''import sys
from pathlib import Path
_root=Path(globals().get("__file__","/kaggle_simulations/agent/main.py")).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0,str(_root))
from _local_economy.runtime import SubmissionAgent
_submission=SubmissionAgent(_root / "_local_economy")
def agent(observation,configuration=None):
    return _submission.act(observation,configuration)
'''
def export(checkpoint_path,output):
    import torch
    torch.set_num_threads(1)
    ck=read_checkpoint(checkpoint_path,torch)
    model=Policy(torch);model.module.load_state_dict(ck["network"]);model.module.eval()
    weights={k:v.detach().cpu().tolist() for k,v in ck["network"].items() if not k.startswith("value.")}
    if any(not torch.isfinite(v).all() for v in ck["network"].values()):raise ValueError("nonfinite weights")
    native=NativePolicy(weights)
    generator=torch.Generator().manual_seed(8471)
    error=0.
    for n in (1,2,6):
        context=torch.randn((1,CONTEXT_SIZE),generator=generator)
        x=torch.randn((1,n,CANDIDATE_SIZE),generator=generator)
        with torch.no_grad():expected,_=model(context,x,torch.ones((1,n),dtype=torch.bool))
        actual=native.probabilities(context[0].tolist(),x[0].tolist())
        error=max(error,max(abs(a-b) for a,b in zip(actual,expected[0].tolist())))
        if error>2e-5 or greedy(actual)!=greedy(expected[0].tolist()):raise ValueError("runtime policy parity failed")
    raw=json.dumps(weights,separators=(",",":"),allow_nan=False).encode()
    metadata=dict(schema=SCHEMA,settings=ck["settings"],engine=ck["engine"],
                  baseline_sha256=ck["baseline_sha256"],weights_sha256=hashlib.sha256(raw).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                  decision_mode="greedy keep/change then alternative",contains_runtime_baseline=True,
                  resource_validation="pending; this exporter checks structure, size and probabilities only")
    files={"main.py":ENTRY.encode(),"_local_economy/baseline.py":BASELINE.read_bytes(),
           "_local_economy/weights.json":raw,"_local_economy/metadata.json":json.dumps(metadata,indent=2).encode(),
           "NOTICE.txt":b"Baseline-preserving local economic policy. Baseline source retains its embedded licenses and notices.\n"}
    for name in ("__init__.py","settings.py","contracts.py","encoding.py","adapter.py","policy.py","runtime.py"):
        files["_local_economy/"+name]=Path(__file__).with_name(name).read_bytes()
    unpacked=sum(len(v) for v in files.values())
    if unpacked>MAX_DISK:raise ValueError("unpacked package exceeds 8 GiB")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False,suffix=".tar.gz") as tmp:path=Path(tmp.name)
    try:
        with tarfile.open(path,"w:gz") as tar:
            for name,data in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(data))
        size=path.stat().st_size
        if size>MAX_ARCHIVE:raise ValueError("archive exceeds 100 MiB")
        path.replace(output)
    finally:path.unlink(missing_ok=True)
    return dict(archive=str(output),compressed_bytes=size,unpacked_bytes=unpacked,
                max_probability_log_error=error,baseline_bundled=True,resource_validation="pending")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.out),indent=2))
if __name__=="__main__":main()
