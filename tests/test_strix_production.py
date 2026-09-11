import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import qwen_strix_service_smoke as smoke

PATH = Path(__file__).resolve().parents[1] / 'deployment/run-strix-production.py'
SPEC = importlib.util.spec_from_file_location('strix_production', PATH)
prod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prod)


class ProductionTests(unittest.TestCase):
    def test_command_matches_validated_trial_except_api_alias_log_level_and_cache(self):
        expected = smoke.command_for(prod.TRIAL, prod.MODELS)
        for flag, value in (('--host', '0.0.0.0'), ('--port', '8080'), ('--cache-ram', '12288'), ('-lv', '3')):
            expected[expected.index(flag) + 1] = value
        expected += ['--alias', prod.ALIAS]
        self.assertEqual(prod.command_for('0.0.0.0', 8080), expected)

    def test_auth_precedence_and_existing_bind_preserved(self):
        args = types.SimpleNamespace(api_key=None, api_key_file=None)
        with mock.patch.dict(os.environ, {'LLAMA_HOST': '192.168.0.242', 'LLAMA_PORT': '8081',
                                         'LLAMA_API_KEY': 'first,second'}, clear=True):
            self.assertEqual(prod.settings(args), ('192.168.0.242', 8081, 'first,second', '', 'first'))
            args.api_key = 'override'
            self.assertEqual(prod.settings(args)[-1], 'override')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=True):
            key_file = Path(tmp) / 'keys'
            key_file.write_text('\nfile-key\nsecond-key\n')
            args = types.SimpleNamespace(api_key=None, api_key_file=str(key_file))
            self.assertEqual(prod.settings(args)[-1], 'file-key')
            key_file.write_text('')
            with self.assertRaises(ValueError):
                prod.settings(args)

    def test_empty_or_conflicting_auth_rejected(self):
        args = types.SimpleNamespace(api_key=None, api_key_file=None)
        for env in ({}, {'LLAMA_API_KEY': ','}, {'LLAMA_API_KEY': 'secret', 'LLAMA_ARG_API_KEY_FILE': '/missing'}):
            with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                prod.settings(args)

    def test_legacy_overrides_do_not_leak_to_runtime(self):
        with mock.patch.dict(os.environ, {'LLAMA_MTP_MODE': 'off', 'LLAMA_CACHE_RAM_MIB': '16384',
                'GGML_CUDA_DISABLE_GRAPHS': '1', 'ROCR_VISIBLE_DEVICES': '0', 'HIP_VISIBLE_DEVICES': '0',
                'LLAMA_API_KEY': 'old', 'API_KEY': 'old', 'XDG_CACHE_HOME': '/var/cache/qwen-flash-next'}, clear=True):
            env = prod.runtime_environment('', '/etc/qwen.keys')
        self.assertEqual(env['LLAMA_ARG_API_KEY_FILE'], '/etc/qwen.keys')
        self.assertNotIn('LLAMA_API_KEY', env)
        self.assertNotIn('API_KEY', env)
        self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
        self.assertNotIn('HIP_VISIBLE_DEVICES', env)
        self.assertNotIn('LLAMA_MTP_MODE', env)
        self.assertNotIn('GGML_CUDA_DISABLE_GRAPHS', env)
        self.assertEqual(env['XDG_CACHE_HOME'], '/var/cache/qwen-flash-next')

    def test_validate_rejects_changed_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = prod.command_for('127.0.0.1', 8080)
            command[0] = str(root / 'llama-server')
            for flag in ('-m', '-md', '--mmproj'):
                path = root / (flag.replace('-', '') + '.gguf')
                path.touch()
                command[command.index(flag) + 1] = str(path)
            fingerprint = []
            for name in ('llama-server', 'libllama-common.so', 'libllama-server-impl.so'):
                path = root / name
                path.write_bytes(name.encode())
                fingerprint.append({'path': str(path), 'sha256': hashlib.sha256(name.encode()).hexdigest(),
                                    'mtp_state_marker': name == 'libllama-common.so'})
            (root / 'binary-fingerprint.json').write_text(json.dumps(fingerprint))
            (root / 'results.json').write_text(json.dumps({'verdict': {'status': 'PASS', 'mtp_state_restore_logged': True},
                'near_full_output_comparison': {'tokens_match': True}}))
            with mock.patch.dict(os.environ, {'STRIX_VALIDATION_DIR': str(root)}), mock.patch.object(os, 'access', return_value=True):
                prod.validate(command)
                (root / 'libllama-common.so').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Binary changed'):
                    prod.validate(command)

    def test_readiness_checks_authenticated_api_and_mtp_without_printing_key(self):
        calls = []
        def respond(base, key, endpoint, body=None, **kwargs):
            calls.append((base, key, endpoint, body))
            if endpoint == '/props':
                return {'modalities': {'vision': True}, 'default_generation_settings': {'n_ctx': 262144}}
            if endpoint == '/v1/models':
                return {'data': [{'id': prod.ALIAS}]}
            if body:
                return {'choices': [{'message': {'content': 'Weather observations help forecasts.'}}], 'timings': {'draft_n': 3}}
            return {}
        with mock.patch.object(prod, 'request', side_effect=respond), contextlib.redirect_stdout(io.StringIO()) as output:
            prod.ready('0.0.0.0', 8080, 'private-key')
        self.assertNotIn('private-key', output.getvalue())
        self.assertTrue(all(c[0] == 'http://127.0.0.1:8080' and c[1] == 'private-key' for c in calls))
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1][-1]['model'], prod.ALIAS)

    def test_check_does_not_exec_or_make_requests(self):
        with mock.patch('sys.argv', ['run-strix-production.py', '--check']), \
                mock.patch.dict(os.environ, {'LLAMA_API_KEY': 'private-key'}, clear=True), \
                mock.patch.object(prod, 'validate'), mock.patch.object(os, 'execve') as execute, \
                mock.patch.object(prod, 'request') as request, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(prod.main(), 0)
        execute.assert_not_called()
        request.assert_not_called()
        self.assertNotIn('private-key', output.getvalue())


if __name__ == '__main__':
    unittest.main()
