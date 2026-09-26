"""Small interface check by default; full identity tests must be requested."""
import argparse,json
from ..paths import BASELINE
from .adapter import load_namespace
from .settings import SCHEMA
from .runner import Serve,run_game
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity-seeds",type=int,nargs="+")
    a=p.parse_args()
    ns,entry,digest=load_namespace(BASELINE)
    report=dict(schema=SCHEMA,baseline_sha256=digest,hook_ready=True,bc_required=False,
                full_identity="not run",economic_validation="not run",resource_validation="not run")
    if a.identity_seeds:
        report["identity_games"]=[]
        with Serve() as server:
            for seed in a.identity_seeds:
                for seat in (0,1):
                    report["identity_games"].append(run_game(server,seed,seat,verify_keep=True))
        report["full_identity"]="passed"
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
