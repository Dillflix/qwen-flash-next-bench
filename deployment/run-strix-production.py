#!/usr/bin/env python3
"""Launch the validated single-gfx1151 deployment; retain existing API settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

TRIAL = Path('/srv/llm/src/strix-llama-trial-5f851647')
MODELS = Path('/srv/llm/models/qwen-flash-next')
EVIDENCE = TRIAL / 'trial-results/hip-near-full-retention.p0ia0m_5'
ALIAS = 'Qwen/Qwen3.8-Flash-Next-think'


def command_for(host, port):
    return [str(TRIAL / 'build-hip10-dual/bin/llama-server'),
        '-m', str(MODELS / 'strix-bf16-joined.wPbjNP/quantized/Qwen3.8-Flash-Next-Q5_K-IQ4_NL-PLE-Q8_0.gguf'),
        '--host', host, '--port', str(port), '--ctx-size', '262144', '--parallel', '1',
        '--device', 'ROCm1', '--split-mode', 'none', '--n-gpu-layers', '999', '--fit', 'off',
        '--load-mode', 'none', '--ngram-on-disk', '--ngram-direct-io', '--ngram-cache', '256',
        '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
        '--batch-size', '2048', '--ubatch-size', '1536', '--threads', '16',
        '--cache-ram', '12288', '--ctx-checkpoints', '8', '--checkpoint-min-step', '32768',
        '--no-kv-unified', '--cache-idle-slots', '--no-context-shift', '--jinja', '-lv', '3',
        '-md', str(MODELS / 'mtp-Qwen3.8-Flash-Next-Q8_0.gguf'),
        '--spec-type', 'draft-mtp', '--spec-draft-device', 'ROCm1', '--spec-draft-ngl', '999',
        '--spec-draft-n-max', '3', '--spec-draft-p-min', '0.75',
        '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16',
        '--mmproj', str(MODELS / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'),
        '--mmproj-offload', '--mmproj-device', 'ROCm1',
        '--image-min-tokens', '1024', '--image-max-tokens', '2240', '--alias', ALIAS]


def settings(args):
    host = os.environ.get('LLAMA_HOST', '127.0.0.1').strip()
    port = int(os.environ.get('LLAMA_PORT', '8080'))
    if not host or not 1 <= port <= 65535:
        raise ValueError('Invalid LLAMA_HOST or LLAMA_PORT')
    direct = os.environ.get('LLAMA_API_KEY') or os.environ.get('QWEN_API_KEY') or os.environ.get('API_KEY', '')
    key_file = os.environ.get('LLAMA_ARG_API_KEY_FILE', '')
    if args.api_key is not None:
        direct, key_file = args.api_key, ''
    elif args.api_key_file is not None:
        direct, key_file = '', args.api_key_file
    if direct and key_file:
        raise ValueError('Configure either an API key or API-key file, not both')
    keys = [key.strip() for key in (Path(key_file).read_text().splitlines() if key_file else direct.split(',')) if key.strip()]
    if not keys:
        raise ValueError('A nonempty API key or API-key file is required for production')
    if any('\n' in key or '\r' in key for key in keys):
        raise ValueError('Invalid API key encoding')
    return host, port, direct, key_file, keys[0]


def validate(command):
    for flag in ('-m', '-md', '--mmproj'):
        path = Path(command[command.index(flag) + 1])
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f'Missing/unreadable {flag} model: {path}')
    if not os.access(command[0], os.X_OK):
        raise ValueError('Validated Strix server is missing or not executable')
    evidence = Path(os.environ.get('STRIX_VALIDATION_DIR', str(EVIDENCE)))
    report = json.loads((evidence / 'results.json').read_text())
    if (report.get('verdict', {}).get('status') != 'PASS'
            or not report.get('verdict', {}).get('mtp_state_restore_logged')
            or not report.get('near_full_output_comparison', {}).get('tokens_match')):
        raise ValueError('Successful near-full MTP retention evidence is required')
    fingerprints = json.loads((evidence / 'binary-fingerprint.json').read_text())
    expected = {str(Path(command[0])), str(Path(command[0]).parent / 'libllama-common.so'),
                str(Path(command[0]).parent / 'libllama-server-impl.so')}
    if {entry['path'] for entry in fingerprints} != expected:
        raise ValueError('Validation fingerprint paths do not match this deployment')
    marked = False
    for entry in fingerprints:
        path = Path(entry['path'])
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Binary changed since retention validation: {path.name}')
        marked |= entry.get('mtp_state_marker', False)
    if not marked:
        raise ValueError('MTP state patch was not recorded in the validated binaries')


def runtime_environment(direct, key_file):
    # Do not let legacy dual-GPU, bypass, graph, cache, or visibility settings
    # silently change the tested runtime. Auth and bind are handled explicitly.
    excluded = {'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                'GPU_DEVICE_ORDINAL', 'QWEN_API_KEY', 'API_KEY'}
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_')) and k not in excluded}
    env['LD_LIBRARY_PATH'] = '/opt/rocm-10.0.0/lib:/opt/rocm-10.0.0/lib64:/opt/rocm-10.0.0/lib/rocm_sysdeps/lib'
    if key_file:
        env['LLAMA_ARG_API_KEY_FILE'] = key_file
    else:
        env['LLAMA_API_KEY'] = direct
    return env


def request(base, key, endpoint, body=None, timeout=10):
    headers = {'Authorization': 'Bearer ' + key}
    data = None
    if body is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(body).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(base + endpoint, data=data, headers=headers), timeout=timeout) as response:
        return json.load(response)


def ready(host, port, key):
    host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)
    if ':' in host and not host.startswith('['):
        host = '[' + host + ']'
    base = f'http://{host}:{port}'
    deadline = time.monotonic() + 600
    while True:
        try:
            request(base, key, '/health')
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Strix server did not become healthy within 600 seconds')
            time.sleep(2)
    props = request(base, key, '/props')
    if (not props.get('modalities', {}).get('vision')
            or props.get('default_generation_settings', {}).get('n_ctx') != 262144):
        raise RuntimeError('Server did not confirm 262144 context and vision')
    models = request(base, key, '/v1/models')
    if not any(item.get('id') == ALIAS or ALIAS in item.get('aliases', []) for item in models.get('data', [])):
        raise RuntimeError('API model alias is missing')
    result = request(base, key, '/v1/chat/completions', {
        'model': ALIAS, 'messages': [{'role': 'user', 'content':
            'Explain in two complete sentences how temperature and wind observations help weather forecasting.'}],
        'max_tokens': 96, 'temperature': 0, 'seed': 1234, 'stream': False,
        'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=180)
    choices = result.get('choices') or []
    if not choices:
        raise RuntimeError('Authenticated startup check returned no completion choice')
    content = choices[0].get('message', {}).get('content') or ''
    if not content.strip() or result.get('timings', {}).get('draft_n', 0) <= 0:
        raise RuntimeError('Authenticated text/MTP startup check failed')
    print('Strix production ready: authenticated API, 256K allocation, vision loaded, MTP drafting confirmed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument('--api-key')
    auth.add_argument('--api-key-file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    host, port, direct, key_file, key = settings(args)
    if args.ready:
        ready(host, port, key)
        return 0
    command = command_for(host, port)
    validate(command)
    print(f'Strix production: {host}:{port}; gfx1151 only; MTP n=3; F16; cache=12288 MiB; PLE on SSD.', flush=True)
    if args.check:
        return 0
    os.execve(command[0], command, runtime_environment(direct, key_file))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(66)
