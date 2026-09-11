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
    def test_mtp_off_removes_draft_options_without_changing_target_or_vision(self):
        on = smoke.command_for(pathlib.Path('/trial'), pathlib.Path('/models'))
        off = smoke.without_mtp(on)
        self.assertNotIn('-md', off)
        self.assertFalse(any(arg.startswith('--spec-') for arg in off))
        for flag in ('-m', '--device', '--mmproj', '--mmproj-device', '--cache-ram', '--ctx-size'):
            self.assertEqual(on[on.index(flag) + 1], off[off.index(flag) + 1])
        self.assertIn('--ngram-on-disk', off)

    def test_repeat_classification(self):
        def response(tokens, cached=0):
            return {'tokens': tokens, 'content': str(tokens),
                    'timings': {'cache_n': cached, 'prompt_n': 32768 - cached, 'draft_n': 5}}
        data = {'uncached_1': response([1, 2]), 'uncached_2': response([1, 2]),
                'cached': response([1, 2], 32764)}
        self.assertEqual(smoke.repeat_verdict(data, 32768, True)['status'], 'PASS')
        data['cached'] = response([1, 3], 32764)
        self.assertEqual(smoke.repeat_verdict(data, 32768, True)['status'], 'CACHE_PATH_DRIFT')
        data['uncached_2'] = response([1, 3])
        self.assertEqual(smoke.repeat_verdict(data, 32768, True)['status'], 'INCONCLUSIVE_UNCACHED_DRIFT')
        data['uncached_1'] = response([1, 2], 5)
        self.assertEqual(smoke.repeat_verdict(data, 32768, True)['status'], 'INVALID')

    def test_missing_draft_activity_invalidates_enabled_arm(self):
        response = {'tokens': [1], 'content': 'one', 'timings': {'cache_n': 0, 'prompt_n': 10}}
        cached = dict(response, timings={'cache_n': 9, 'prompt_n': 1})
        data = {'uncached_1': response, 'uncached_2': response, 'cached': cached}
        self.assertEqual(smoke.repeat_verdict(data, 10, False)['status'], 'PASS')
        self.assertEqual(smoke.repeat_verdict(data, 10, True)['status'], 'INVALID')

    def test_comparison_detects_shorter_output(self):
        result = smoke.output_comparison({'tokens': [1, 2]}, {'tokens': [1]})
        self.assertFalse(result['tokens_match'])
        self.assertEqual(result['first_divergence_zero_based'], 1)
        self.assertIsNone(result['second_token_id'])

    def test_repeat_probe_request_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def respond(endpoint, body):
                calls.append(body)
                cached = 9 if body['cache_prompt'] else 0
                return {'content': 'valid result', 'tokens': [10, 11],
                        'timings': {'cache_n': cached, 'prompt_n': 10-cached, 'predicted_n': 2, 'draft_n': 1}}
            results = {}
            with mock.patch.object(smoke, 'post', side_effect=respond), \
                    contextlib.redirect_stdout(io.StringIO()):
                smoke.repeat_probes(pathlib.Path(tmp), list(range(10)), results, True)
            self.assertEqual([body['cache_prompt'] for body in calls], [False, False, True])
            self.assertTrue(all('id_slot' not in body for body in calls))
            self.assertEqual(results['verdict']['status'], 'PASS')

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
                args = types.SimpleNamespace(trial_dir=root, model_dir=root, run=True, repeat_ab=False)

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

    def test_ab_continues_after_drift_and_restores_service_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            command = smoke.command_for(root, root)
            for flag in ('-m', '-md', '--mmproj'):
                path = pathlib.Path(command[command.index(flag) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            baseline = root / 'trial-results/hip-templated-32k.xhe6lmo5/prompt-tokens.json'
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps([1] * 32768))
            args = types.SimpleNamespace(trial_dir=root, model_dir=root, run=True, repeat_ab=True)

            def probes(out, tokens, results, mtp):
                results['verdict'] = {'status': 'CACHE_PATH_DRIFT' if mtp else 'PASS'}
                results['uncached_2'] = {'tokens': [1, 2], 'content': 'two'}

            with mock.patch.object(smoke.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    mock.patch.object(smoke.sys, 'platform', 'linux'), \
                    mock.patch.object(smoke.os, 'access', return_value=True), \
                    mock.patch.object(smoke.subprocess, 'check_output', return_value=smoke.PIN), \
                    mock.patch.object(smoke.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run, \
                    mock.patch.object(smoke.subprocess, 'Popen') as launch, \
                    mock.patch.object(smoke, 'memory_monitor'), \
                    mock.patch.object(smoke, 'wait_ready'), \
                    mock.patch.object(smoke, 'stop_server'), \
                    mock.patch.object(smoke, 'repeat_probes', side_effect=probes) as probe, \
                    mock.patch.object(smoke.urllib.request, 'urlopen') as urlopen, \
                    mock.patch.object(smoke.socket, 'socket') as socket, \
                    contextlib.redirect_stdout(io.StringIO()):
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                urlopen.return_value.__enter__.return_value.read.return_value = json.dumps({
                    'modalities': {'vision': True}, 'default_generation_settings': {'n_ctx': 262144}}).encode()
                self.assertEqual(smoke.main(), 1)
            self.assertEqual(launch.call_count, 2)
            self.assertIn('-md', launch.call_args_list[0].args[0])
            self.assertNotIn('-md', launch.call_args_list[1].args[0])
            self.assertEqual([call.kwargs['mtp'] for call in probe.call_args_list], [True, False])
            actions = [call.args[0] for call in run.call_args_list]
            self.assertEqual(actions.count(['sudo', 'systemctl', 'stop', smoke.SERVICE]), 1)
            self.assertEqual(actions.count(['sudo', 'systemctl', 'start', smoke.SERVICE]), 1)
            self.assertEqual(len(list((root / 'trial-results').glob('hip-cache-repeat-ab.*.tar.gz'))), 1)


if __name__ == '__main__':
    unittest.main()
