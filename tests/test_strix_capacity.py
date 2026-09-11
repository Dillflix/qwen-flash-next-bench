import contextlib
import io
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import qwen_strix_capacity as capacity
import qwen_strix_service_smoke as smoke


class CapacityTests(unittest.TestCase):
    def test_exact_budget_and_positions(self):
        tokens, positions = capacity.exact_tokens([1, 2], [9], [3, 4], 30,
                                                  [(8, [5, 6]), (24, [7])])
        self.assertEqual(len(tokens), 30)
        self.assertEqual(positions, [8, 24])
        self.assertEqual(tokens[8:10], [5, 6])
        self.assertEqual(tokens[-1], 9)
        with self.assertRaises(ValueError):
            capacity.exact_tokens([1], [9], [2], 10, [(8, [3, 4, 5])])

    def test_saturation_requires_observed_eviction_and_original_cap(self):
        log = 'cache state: 33 prompts, 8010.000 MiB (limits: 8192.000 MiB, 100 tokens)'
        self.assertFalse(capacity.saturated(capacity.cache_snapshot(log)))
        self.assertTrue(capacity.saturated(capacity.cache_snapshot(capacity.EVICTION + log)))
        self.assertFalse(capacity.saturated(capacity.cache_snapshot(capacity.EVICTION + log.replace('8192.000', '8100.000'))))
        self.assertFalse(capacity.saturated(capacity.cache_snapshot(capacity.EVICTION + log.replace('8010.000', '1000.000'))))
        self.assertIsNone(capacity.cache_snapshot('no occupancy evidence'))

    def test_guard_thresholds(self):
        row = {'SwapFree_GiB': 7.8}
        self.assertIsNone(capacity.guard_reason(row, 8, 2))
        self.assertIn('RAM', capacity.guard_reason(row, 8, 3))
        self.assertIn('Swap', capacity.guard_reason({'SwapFree_GiB': 7.7}, 8, 0))

    def test_capacity_plan_is_read_only_and_excludes_ab(self):
        with mock.patch('sys.argv', ['smoke', '--capacity-validation']), \
                mock.patch.object(smoke.subprocess, 'run') as run, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(smoke.main(), 0)
        run.assert_not_called()
        self.assertIn('253952', output.getvalue())
        with mock.patch('sys.argv', ['smoke', '--capacity-validation', '--repeat-ab']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main()

    def exercise(self, *, admission_full=True, retrieval=True, saturate=True, trip=False):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            log = out / 'server.log'
            log.touch()
            result, calls = {}, []
            count = 0

            def fixture(tag, total, needles=False):
                return [1] * total, {'input_tokens': total}

            def post(endpoint, body, **kwargs):
                nonlocal count
                count += 1
                calls.append(body)
                full = count >= 4 and saturate
                long = len(body['prompt']) == capacity.NEAR_FULL
                if long and not admission_full:
                    full = False
                with log.open('a') as sink:
                    if full:
                        sink.write(capacity.EVICTION + '\n')
                    sink.write(f'cache state: 33 prompts, {8010 if full else 500}.000 MiB (limits: 8192.000 MiB, 100 tokens)\n')
                    if body['cache_prompt']:
                        sink.write('found better prompt with f_keep\n' + capacity.STATE_MARKER.decode() + '\n')
                    if len(calls) > 1 and len(calls[-2]['prompt']) == capacity.NEAR_FULL:
                        sink.write('prompt state size exceeds cache size limit, skipping\n')
                if trip and long:
                    result['capacity_guard'].update(tripped=True, reason='synthetic memory guard')
                return {'tokens': [10, 11], 'content': ' '.join(capacity.EXPECTED) if long and retrieval else 'Weather observations help.',
                        'timings': {'predicted_n': 2, 'draft_n': 1,
                                    'cache_n': 252 if body['cache_prompt'] else 0,
                                    'prompt_n': 4 if body['cache_prompt'] else len(body['prompt'])}}

            with mock.patch.object(capacity, 'fixture', side_effect=fixture), \
                    mock.patch.object(capacity, 'post', side_effect=post), \
                    mock.patch.object(capacity, 'read_memory', return_value={'SwapFree_GiB': 8, 'MemAvailable_GiB': 16}), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    capacity.capacity_probes(out, result, mock.Mock())
                except RuntimeError as error:
                    result['exception'] = str(error)
            return result, calls

    def test_single_long_request_after_saturation_and_restore(self):
        result, calls = self.exercise()
        self.assertEqual(result['verdict']['status'], 'PASS')
        self.assertTrue(result['near_full_state_exceeded_cache_cap'])
        long = [body for body in calls if len(body['prompt']) == capacity.NEAR_FULL]
        self.assertEqual(len(long), 1)
        self.assertEqual(long[0]['n_predict'], 128)
        self.assertFalse(long[0]['cache_prompt'])
        self.assertTrue(any(body['cache_prompt'] for body in calls))
        self.assertTrue(all('id_slot' not in body for body in calls))

    def test_does_not_start_long_request_without_saturation(self):
        result, calls = self.exercise(saturate=False)
        self.assertIn('saturation not demonstrated', result['exception'])
        self.assertEqual(len(calls), 64)
        self.assertFalse(any(len(body['prompt']) == capacity.NEAR_FULL for body in calls))

    def test_long_admission_must_stay_occupied(self):
        result, _ = self.exercise(admission_full=False)
        self.assertEqual(result['verdict']['status'], 'FAIL')
        self.assertIn('occupancy', result['verdict']['failures'][0])

    def test_retrieval_failure_not_capacity_pass(self):
        result, _ = self.exercise(retrieval=False)
        self.assertEqual(result['verdict']['status'], 'FAIL')

    def test_guard_cannot_report_pass_after_response(self):
        result, _ = self.exercise(trip=True)
        self.assertIn('synthetic memory guard', result['exception'])
        self.assertNotIn('verdict', result)

    def test_main_capacity_failure_archives_restores_and_needs_no_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / 'trial-results').mkdir()
            command = smoke.command_for(root, root)
            for flag in ('-m', '-md', '--mmproj'):
                path = pathlib.Path(command[command.index(flag) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            args = types.SimpleNamespace(trial_dir=root, model_dir=root, run=True,
                                         repeat_ab=False, capacity_validation=True)
            with mock.patch.object(smoke.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    mock.patch.object(smoke.sys, 'platform', 'linux'), \
                    mock.patch.object(smoke.os, 'access', return_value=True), \
                    mock.patch.object(smoke, 'state_patch_artifacts', return_value=[{'mtp_state_marker': True}]), \
                    mock.patch.object(smoke.subprocess, 'check_output', return_value=smoke.PIN), \
                    mock.patch.object(smoke.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run, \
                    mock.patch.object(smoke.subprocess, 'Popen'), \
                    mock.patch.object(smoke, 'memory_monitor'), \
                    mock.patch.object(smoke, 'wait_ready'), \
                    mock.patch.object(smoke, 'stop_server') as stop, \
                    mock.patch.object(capacity, 'capacity_probes', side_effect=RuntimeError('synthetic capacity failure')) as probe, \
                    mock.patch.object(smoke, 'probes') as old_probe, \
                    mock.patch.object(smoke.urllib.request, 'urlopen') as urlopen, \
                    mock.patch.object(smoke.socket, 'socket') as socket, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                urlopen.return_value.__enter__.return_value.read.return_value = json.dumps({
                    'modalities': {'vision': True}, 'default_generation_settings': {'n_ctx': 262144}}).encode()
                self.assertEqual(smoke.main(), 1)
            probe.assert_called_once()
            old_probe.assert_not_called()
            stop.assert_called_once()
            actions = [call.args[0] for call in run.call_args_list]
            self.assertEqual(actions.count(['sudo', 'systemctl', 'start', smoke.SERVICE]), 1)
            self.assertEqual(len(list((root / 'trial-results').glob('hip-cache-full-context.*.tar.gz'))), 1)


if __name__ == '__main__':
    unittest.main()
