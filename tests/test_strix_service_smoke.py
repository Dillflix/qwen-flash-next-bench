import contextlib
import io
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import qwen_strix_service_smoke as smoke


class ServiceSmokeTests(unittest.TestCase):
    def test_device_and_memory_constraints(self):
        command = smoke.command_for(pathlib.Path('/trial'), pathlib.Path('/models'))
        for option, value in {
            '--device': 'ROCm1', '--spec-draft-device': 'ROCm1', '--mmproj-device': 'ROCm1',
            '--ctx-size': '262144', '--parallel': '1', '--cache-ram': '8192',
            '--load-mode': 'none', '--ctx-checkpoints': '8', '--checkpoint-min-step': '32768',
            '--cache-type-k': 'f16', '--cache-type-v': 'f16',
            '--spec-draft-type-k': 'f16', '--spec-draft-type-v': 'f16', '--spec-draft-n-max': '3',
        }.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn('--ngram-on-disk', command)
        self.assertIn('--ngram-direct-io', command)
        self.assertNotIn('--mmap', command)

    def test_common_prefix(self):
        self.assertEqual(smoke.lcp([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(smoke.lcp([1], [1, 2]), 1)
        self.assertEqual(smoke.lcp([], [1]), 0)

    def test_degenerate_response_is_not_success(self):
        for value in ({}, {'content': 'x', 'timings': {'predicted_n': 1}},
                      {'content': 'x', 'timings': {'predicted_n': 32}}):
            with self.assertRaises(ValueError):
                smoke.text_result(value)

    def test_cache_reuse_must_exceed_diversion_prefix(self):
        first = {'content': 'text', 'tokens': [5, 6]}
        later = dict(first, timings={'cache_n': 12})
        self.assertFalse(smoke.compare_cache(first, later, 12)['reused_beyond_current_prefix'])
        self.assertTrue(smoke.compare_cache(first, later, 11)['reused_beyond_current_prefix'])

    def test_missing_evidence_fails(self):
        self.assertEqual(smoke.evaluate({})['status'], 'FAIL')

    def test_clean_smoke_passes_only_with_restore_evidence(self):
        comparison = {'reused_beyond_current_prefix': True, 'output_tokens_match': True,
                      'output_text_matches': True}
        results = {'cache_summary': {'live': comparison, 'after_b': comparison,
                                    'backing_restore_selection_logged': True},
                   'vision_summary': {'content': 'good fixture', 'missing_anchors': []},
                   'a_cold': {'timings': {'draft_n': 20}}}
        self.assertEqual(smoke.evaluate(results)['status'], 'PASS')
        results['cache_summary']['backing_restore_selection_logged'] = False
        self.assertEqual(smoke.evaluate(results)['status'], 'FAIL')

    def test_probe_sequence_uses_automatic_slot_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            (out / 'server.log').touch()
            calls = []
            def respond(endpoint, body):
                calls.append((endpoint, body))
                if endpoint == '/apply-template':
                    return {'prompt': 'prime'}
                if endpoint == '/tokenize':
                    return {'tokens': [1, 9]}
                if endpoint == '/completion':
                    if len([c for c in calls if c[0] == endpoint]) == 4:
                        with (out / 'server.log').open('a') as log:
                            log.write('found better prompt with f_keep = 1.0')
                    return {'content': 'valid result', 'tokens': [10, 11],
                            'timings': {'cache_n': 100, 'predicted_n': 2, 'draft_n': 1}}
                return {'choices': [{'message': {'content':
                    'red circle blue square green triangle UNSLOTH 42'}}], 'timings': {'draft_n': 1}}
            results = {}
            with mock.patch.object(smoke, 'post', side_effect=respond), \
                    contextlib.redirect_stdout(io.StringIO()):
                smoke.probes(out, [1, 2, 3], results)
            completions = [body for endpoint, body in calls if endpoint == '/completion']
            self.assertEqual(len(completions), 4)
            self.assertTrue(all('id_slot' not in body for body in completions))
            self.assertTrue(all(body['cache_prompt'] and body['return_tokens'] for body in completions))
            self.assertEqual(smoke.evaluate(results)['status'], 'PASS')
            self.assertTrue((out / 'vision-summary.json').is_file())

    def test_environment_isolation_does_not_log_values(self):
        with mock.patch.dict(smoke.os.environ, {'PATH': '/bin', 'LLAMA_API_KEY': 'private',
                'GGML_CUDA_DISABLE_GRAPHS': '1', 'HIP_VISIBLE_DEVICES': '0'}, clear=True):
            env, removed = smoke.trial_environment()
        self.assertNotIn('LLAMA_API_KEY', env)
        self.assertNotIn('HIP_VISIBLE_DEVICES', env)
        self.assertNotIn('private', json.dumps(removed))
        self.assertEqual(env['PATH'], '/bin')

    def test_default_plan_does_not_touch_services(self):
        with mock.patch('sys.argv', ['qwen_strix_service_smoke.py']), \
                mock.patch.object(smoke.subprocess, 'run') as run, \
                mock.patch.object(smoke.subprocess, 'check_output') as check, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(), 0)
        run.assert_not_called()
        check.assert_not_called()

    def test_launch_failure_archives_and_restores_only_previously_active_service(self):
        for active in (True, False):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                command = smoke.command_for(root, root)
                for flag in ('-m', '-md', '--mmproj'):
                    path = pathlib.Path(command[command.index(flag) + 1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                baseline = root / 'trial-results/hip-templated-32k.xhe6lmo5/prompt-tokens.json'
                baseline.parent.mkdir(parents=True)
                baseline.write_text(json.dumps([1] * 32768))
                args = types.SimpleNamespace(trial_dir=root, model_dir=root, run=True)

                def run_result(command, **kwargs):
                    code = (0 if active else 3) if command[:2] == ['systemctl', 'is-active'] else 0
                    return types.SimpleNamespace(returncode=code)

                with mock.patch.object(smoke.argparse.ArgumentParser, 'parse_args', return_value=args), \
                        mock.patch.object(smoke.sys, 'platform', 'linux'), \
                        mock.patch.object(smoke.os, 'access', return_value=True), \
                        mock.patch.object(smoke.subprocess, 'check_output', return_value=smoke.PIN), \
                        mock.patch.object(smoke.subprocess, 'run', side_effect=run_result) as run, \
                        mock.patch.object(smoke.subprocess, 'Popen', side_effect=OSError('test launch failure')), \
                        mock.patch.object(smoke, 'memory_monitor'), \
                        mock.patch.object(smoke.socket, 'socket') as socket, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                    self.assertEqual(smoke.main(), 1)
                actions = [call.args[0] for call in run.call_args_list]
                self.assertEqual(['sudo', 'systemctl', 'start', smoke.SERVICE] in actions, active)
                self.assertEqual(['sudo', 'systemctl', 'stop', smoke.SERVICE] in actions, active)
                archives = list((root / 'trial-results').glob('hip-cache-vision-256k.*.tar.gz'))
                self.assertEqual(len(archives), 1)
                self.assertTrue(pathlib.Path(str(archives[0]) + '.sha256').is_file())


if __name__ == '__main__':
    unittest.main()
