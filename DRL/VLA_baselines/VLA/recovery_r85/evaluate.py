"""R83 evaluator unchanged except explicit JSON metadata Path conversion."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json as original_atomic_json

def json_metadata(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:json_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_metadata(v) for v in value]
    return value

def atomic_json(path,value):
    return original_atomic_json(path,json_metadata(value))

def main(a):
    # Process-local dependency injection: frozen historical source is untouched.
    import VLA.autonomy_r83.evaluate as old
    old.atomic_json=atomic_json
    old.main(a)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('method','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
