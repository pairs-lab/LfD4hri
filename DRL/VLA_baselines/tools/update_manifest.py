"""Refresh reviewed package hashes while retaining original export hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','__pycache__','third_party','.venv','.venvs','.pytest_cache','runs','validation'}
ALLOWED={'.py','.json','.md','.txt','.sh','.patch','.toml','.yaml','.yml','.cfg','.ini','.xml','.urdf','.obj','.stl','.dae','.png','.jpg'}


def main():
    path=ROOT/'SOURCE_MANIFEST.json';old=json.loads(path.read_text());prior={r['path']:r for r in old['files']}
    rows=[]
    for file in sorted(ROOT.rglob('*')):
        rel=file.relative_to(ROOT)
        if (any(p in IGNORED for p in rel.parts) or file.is_symlink() or not file.is_file()
            or file.name in ('SOURCE_MANIFEST.json','PACKAGE_CHECK.json')
            or (file.suffix.lower() not in ALLOWED and file.name!='.gitignore')):continue
        row=dict(prior.get(str(rel),{}));row['path']=str(rel)
        sha=hashlib.sha256(file.read_bytes()).hexdigest()
        if row.get('sha256') and row['sha256']!=sha:row.setdefault('original_export_sha256',row['sha256'])
        row.update(sha256=sha,bytes=file.stat().st_size);rows.append(row)
    old.update(source_scope='VLA_baselines source and robot21 adapters; original DRL/VLA is unchanged',files=rows)
    path.write_text(json.dumps(old,indent=2)+'\n');print('Source manifest updated:',len(rows),'files')


if __name__=='__main__':main()
