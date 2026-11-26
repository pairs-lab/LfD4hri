"""Validate this source export without loading weights or running model jobs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.onnx', '.npz', '.npy',
             '.h5', '.hdf5', '.parquet', '.arrow', '.mp4', '.webm', '.avi', '.mov',
             '.zip', '.tar', '.gz', '.pyc', '.pyo'}
IGNORED_DIRS = {'.git', '__pycache__', 'third_party', '.venv', '.venvs', '.pytest_cache'}


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Current package source missing/modified: ' + row['path'])
    checked = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            if str(relative) != 'asset':
                errors.append('Unexpected symlink: ' + str(relative))
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN or path.stat().st_size > 5 * 1024 * 1024:
            errors.append('Unexpected weight/data/large artifact: ' + str(relative))
        if path.name == '.env' or path.suffix in {'.pem', '.key'}:
            errors.append('Sensitive filename: ' + str(relative))
        if path.suffix in {'.py', '.json', '.patch', '.sh', '.toml', '.yaml', '.yml', '.txt', '.md'}:
            text = path.read_text(errors='replace')
            # Report locations only, never credential values.
            if re.search(r'(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)', text):
                errors.append('Possible credential; inspect before upload: ' + str(relative))
        if path.suffix == '.py':
            checked += 1
            try:
                ast.parse(path.read_bytes(), filename=str(relative))
            except SyntaxError as exc:
                errors.append(f'Syntax error: {relative}:{exc.lineno}')
    upstreams = json.loads((ROOT / 'UPSTREAMS.json').read_text())['repositories']
    for model in json.loads((ROOT / 'MODELS.json').read_text())['models']:
        for relative in model['source']:
            if not (ROOT / relative).is_dir():
                errors.append('Model source-map directory missing: ' + relative)
    for row in upstreams:
        if row.get('patch'):
            record = row['patch']
            if hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() != record['sha256']:
                errors.append('Modified upstream patch: ' + row['name'])
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                            'test_state21_reference.py'], cwd=ROOT / 'protocol21',
                           capture_output=True, text=True)
    print(tests.stdout + tests.stderr)
    if tests.returncode:
        errors.append('21-D numerical contract tests failed; install numpy first')
    integration = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','protocol21/tests','-v'],
                                 cwd=ROOT,capture_output=True,text=True)
    print(integration.stdout+integration.stderr)
    if integration.returncode:
        errors.append('Robot21 integration tests failed; install base test dependencies')
    result = {'source_hashes': len(manifest['files']), 'python_syntax_files': checked,
              'upstream_repositories': len(upstreams), 'errors': errors,
              'native_model_inference_tested': False, 'training_started': False}
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
