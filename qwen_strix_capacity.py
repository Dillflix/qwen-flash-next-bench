"""Occupied backing-cache inference and focused near-full state retention trials."""
import hashlib
import json
import pathlib
import re
import threading
import time

from qwen_strix_service_smoke import post, write_json, text_result, output_comparison, STATE_MARKER

CAP_MIB = 8192
NEAR_FULL = 253952
CACHE_RE = re.compile(r'cache state: (\d+) prompts, ([\d.]+) MiB \(limits: ([\d.]+) MiB')
EVICTION = 'making room for prompt cache entry, removing oldest entry'
MARKER = 'STRIX_CAPACITY_BODY_87A21'
EXPECTED = ['EMBER-731942', 'CEDAR-482615', 'ORBIT-906273']
RETENTION_CAP_MIB = 12288
RESTORE_TIMEOUT = 180


def cache_snapshot(log):
    events = list(CACHE_RE.finditer(log))
    if not events:
        return None
    event = events[-1]
    count, used, cap = int(event[1]), float(event[2]), float(event[3])
    return {'entries': count, 'used_MiB': used, 'cap_MiB': cap,
            'occupied_percent': 100 * used / cap if cap else 0,
            'capacity_evictions': log.count(EVICTION)}


def saturated(snapshot):
    return bool(snapshot and snapshot['cap_MiB'] == CAP_MIB
                and 95 <= snapshot['occupied_percent'] <= 100.001
                and snapshot['capacity_evictions'] > 0)


def exact_tokens(prefix, suffix, filler, total, inserts=()):
    if not filler or len(prefix) + len(suffix) > total:
        raise ValueError('Invalid exact-token fixture budget')
    result, positions = list(prefix), []
    def pad_to(position):
        gap = position - len(result)
        if gap < 0:
            raise ValueError('Overlapping fixture inserts')
        result.extend((filler * ((gap + len(filler) - 1) // len(filler)))[:gap])
    for position, value in inserts:
        pad_to(position)
        positions.append(len(result))
        result.extend(value)
    pad_to(total - len(suffix))
    result.extend(suffix)
    if len(result) != total:
        raise ValueError('Exact-token fixture length mismatch')
    return result, positions


def tokenize(text):
    tokens = post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']
    if not tokens or not all(type(token) is int for token in tokens):
        raise ValueError('Tokenizer did not return nonempty integer token IDs')
    return tokens


def fixture(tag, total, needles=False):
    task = ('Return the EARLY, MIDDLE and LATE ledger codes, exactly as written, in that order. '
            'Do not invent codes or add an explanation.' if needles else
            'Reply in one complete sentence about weather observations.')
    content = f'{tag}\nRead the reference below.\n{MARKER}\n{task}'
    rendered = post('/apply-template', {'messages': [{'role': 'user', 'content': content}],
                                      'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    if rendered.count(MARKER) != 1:
        raise ValueError('Template did not preserve the fixture marker')
    before, after = rendered.split(MARKER)
    prefix, suffix = tokenize(before), tokenize(after)
    filler = tokenize(' The observatory records temperature, wind and rainfall. These are ordinary weather observations.\n')
    inserts = []
    if needles:
        for position, label, value in zip((1024, 126976, 249000), ('EARLY', 'MIDDLE', 'LATE'), EXPECTED):
            inserts.append((position, tokenize(f'\n{label} ledger code: {value}\n')))
    tokens, positions = exact_tokens(prefix, suffix, filler, total, inserts)
    return tokens, {'tag': tag, 'input_tokens': len(tokens), 'needle_positions': positions,
                    'expected_codes': EXPECTED if needles else [],
                    'token_ids_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}


def read_memory():
    row = {'timestamp': time.time()}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapFree'):
            row[key + '_GiB'] = int(value.split()[0]) / 1048576
    return row


def guard_reason(row, baseline_swap, low_samples):
    if low_samples >= 3:
        return 'Available RAM stayed below 2 GiB for three samples'
    if baseline_swap - row['SwapFree_GiB'] > 0.25:
        return 'Swap use grew by more than 256 MiB during the trial'
    return None


def log_since(path, offset):
    with path.open('rb') as handle:
        handle.seek(offset)
        return handle.read().decode(errors='replace')


def retention_probes(out, results, request, phase):
    """One cold long prefill; a cache miss on return gets only 180 seconds."""
    log = out / 'server.log'
    phase['name'] = 'near_full_cold'
    tokens, manifest = fixture('5e918a30d27f_full_context', NEAR_FULL, needles=True)
    write_json(out / 'near-full-fixture.json', manifest)
    cold = request('near_full_cold', tokens, False, count=128, timeout=7200)
    results['near_full_cold'] = cold
    timing = cold['timings']
    if timing.get('cache_n') != 0 or timing.get('prompt_n') != NEAR_FULL:
        raise RuntimeError('Not an exact uncached 253952-token prefill')
    if timing.get('draft_n', 0) <= 0 or not all(code in cold.get('content', '') for code in EXPECTED):
        raise RuntimeError('Cold near-full response did not pass retrieval with MTP active')

    phase['name'] = 'near_full_save_and_diversion'
    offset = log.stat().st_size
    diversion, _ = fixture('9ba0f714c5d2_post_long', 256)
    results['diversion'] = request('diversion', diversion, False)
    diversion_timing = results['diversion']['timings']
    if diversion_timing.get('cache_n') != 0 or diversion_timing.get('prompt_n') != 256:
        raise RuntimeError('The intervening conversation was not a distinct uncached 256-token prefill')
    save_log = log_since(log, offset)
    saved = cache_snapshot(save_log)
    results['cache_after_near_full_save'] = saved
    save_lengths = [int(value) for value in re.findall(r'saving prompt with length (\d+)', save_log)]
    if ('exceeds cache size limit' in save_log or not saved or saved['cap_MiB'] != RETENTION_CAP_MIB
            or saved['used_MiB'] <= 0 or saved['used_MiB'] > RETENTION_CAP_MIB
            or not any(length >= NEAR_FULL for length in save_lengths)):
        raise RuntimeError('Near-full state retention in the unchanged 12 GiB cache was not demonstrated; not attempting return')

    phase['name'] = 'near_full_backing_restore'
    offset = log.stat().st_size
    try:
        returned = request('near_full_return', tokens, True, count=128, timeout=RESTORE_TIMEOUT)
    except TimeoutError as error:
        raise RuntimeError('Near-full restore exceeded 180 seconds; no retry or second cold prefill will be attempted') from error
    results['near_full_return'] = returned
    restore_log = log_since(log, offset)
    comparison = output_comparison(cold, returned)
    results['near_full_output_comparison'] = comparison
    timing = returned['timings']
    selected = 'found better prompt with f_keep' in restore_log
    state_hook = STATE_MARKER.decode() in restore_log
    errors = []
    if not comparison['tokens_match'] or not comparison['text_matches']:
        errors.append('Restored near-full output differs from the cold output')
    if (timing.get('cache_n', 0) < NEAR_FULL - 2048
            or timing.get('cache_n', 0) + timing.get('prompt_n', 0) != NEAR_FULL):
        errors.append('Near-full reuse was not demonstrated with at most 2048 replayed tokens')
    if not selected or not state_hook:
        errors.append('Backing selection and patched MTP state restore were not both observed')
    if timing.get('draft_n', 0) <= 0:
        errors.append('MTP drafting was not observed after restoration')
    results['verdict'] = {'status': 'FAIL' if errors else 'PASS', 'failures': errors,
        'backing_restore_selection_logged': selected, 'mtp_state_restore_logged': state_hook,
        'scope': 'One near-full cold/save/diversion/restore sequence with a 12 GiB backing-cache limit; not saturation or endurance validation'}


def capacity_probes(out, results, server, *, retention=False):
    log_path = out / 'server.log'
    phase = {'name': 'cache_fill'}
    stop = threading.Event()
    guard = {'tripped': False, 'reason': None}
    results['capacity_guard'] = guard
    baseline_swap = read_memory()['SwapFree_GiB']

    def watch():
        low, next_report = 0, 0
        try:
            with (out / 'capacity-memory.jsonl').open('w', encoding='utf-8') as sink:
                while not stop.is_set():
                    row = read_memory()
                    row['phase'] = phase['name']
                    sink.write(json.dumps(row) + '\n')
                    sink.flush()
                    low = low + 1 if row['MemAvailable_GiB'] < 2 else 0
                    reason = guard_reason(row, baseline_swap, low)
                    if reason:
                        guard.update(tripped=True, reason=reason)
                        print('CAPACITY GUARD: ' + reason, flush=True)
                        if server.poll() is None:
                            server.terminate()
                        return
                    if time.monotonic() >= next_report:
                        print(f"{phase['name']}: available RAM {row['MemAvailable_GiB']:.2f} GiB", flush=True)
                        next_report = time.monotonic() + 30
                    stop.wait(1)
        except Exception as error:
            guard.update(tripped=True, reason=f'Memory monitor failed: {error}')
            if server.poll() is None:
                server.terminate()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    def request(label, tokens, cache, count=16, timeout=300):
        if guard['tripped']:
            raise RuntimeError(guard['reason'])
        body = {'prompt': tokens, 'n_predict': count, 'temperature': 0, 'seed': 1234,
                'cache_prompt': cache, 'return_tokens': True, 'stream': False}
        write_json(out / f'{label}-request.json', body)
        print(f'{label}: {len(tokens)} input tokens, up to {count} output tokens', flush=True)
        started = time.monotonic()
        response = post('/completion', body, timeout=timeout)
        elapsed = time.monotonic() - started
        results.setdefault('request_wall_seconds', {})[label] = elapsed
        write_json(out / f'{label}-response.json', response)
        if guard['tripped']:
            raise RuntimeError(guard['reason'])
        text_result(response)
        print(json.dumps({'label': label, 'wall_seconds': elapsed, 'timings': response.get('timings'),
                          'content': response.get('content')}), flush=True)
        return response

    try:
        if retention:
            retention_probes(out, results, request, phase)
            stop.set()
            watcher.join(timeout=3)
            if guard['tripped'] or watcher.is_alive():
                results.pop('verdict', None)
                raise RuntimeError(guard['reason'] or 'Memory guard did not stop cleanly')
            write_json(out / 'retention-summary.json', results)
            print(json.dumps(results['verdict'], indent=2), flush=True)
            return
        fill = []
        history = []
        for index in range(64):
            tag = hashlib.sha256(f'strix-cache-branch-{index}'.encode()).hexdigest()[:24]
            tokens, manifest = fixture(tag, 256)
            response = request(f'fill_{index:02}', tokens, False)
            timing = response['timings']
            if timing.get('cache_n') != 0 or timing.get('prompt_n') != 256:
                raise RuntimeError('Cache filler did not receive a full independent prefill')
            fill.append((tokens, response))
            snapshot = cache_snapshot(log_path.read_text(errors='replace'))
            history.append({'branch': index, 'cache': snapshot, 'fixture': manifest})
            write_json(out / 'cache-fill-history.json', history)
            print('Backing cache: ' + json.dumps(snapshot), flush=True)
            if saturated(snapshot):
                break
        else:
            raise RuntimeError('Cache saturation not demonstrated within 64 filler requests')

        # Exercise a recently saved entry after capacity eviction, then refill its vacated place.
        phase['name'] = 'saturated_cache_restore'
        if len(fill) < 3:
            raise RuntimeError('Too few distinct branches to verify backing restoration')
        tokens, original = fill[-3]
        offset = log_path.stat().st_size
        restored = request('saturated_restore', tokens, True)
        with log_path.open('rb') as handle:
            handle.seek(offset)
            restore_log = handle.read().decode(errors='replace')
        comparison = output_comparison(original, restored)
        results['saturated_restore'] = comparison
        if (not comparison['tokens_match'] or not comparison['text_matches']
                or restored['timings'].get('cache_n', 0) <= 0
                or 'found better prompt with f_keep' not in restore_log
                or STATE_MARKER.decode() not in restore_log):
            raise RuntimeError('Saturated backing-cache restore did not preserve output and execute the patched state hook')
        refill, _ = fixture('f73c09d814a6c12d_refill', 256)
        request('refill', refill, False)
        before = cache_snapshot(log_path.read_text(errors='replace'))
        results['cache_before_long'] = before
        if not saturated(before):
            raise RuntimeError('Backing cache was not saturated immediately before the long request')

        phase['name'] = 'near_full_prefill_and_decode'
        tokens, manifest = fixture('5e918a30d27f_full_context', NEAR_FULL, needles=True)
        write_json(out / 'near-full-fixture.json', manifest)
        long_offset = log_path.stat().st_size
        near = request('near_full', tokens, False, count=128, timeout=7200)
        results['near_full'] = near
        with log_path.open('rb') as handle:
            handle.seek(long_offset)
            long_log = handle.read().decode(errors='replace')
        # The server reports occupancy after saving/loading on admission, before prefill.
        during = cache_snapshot(long_log)
        results['cache_at_long_admission'] = during
        timing = near['timings']
        errors = []
        if not during or during['cap_MiB'] != CAP_MIB or not 95 <= during['occupied_percent'] <= 100.001:
            errors.append('At least 95% backing-cache occupancy during long inference was not demonstrated')
        if timing.get('cache_n') != 0 or timing.get('prompt_n') != NEAR_FULL:
            errors.append('Not an exact uncached 253952-token prefill')
        if timing.get('draft_n', 0) <= 0:
            errors.append('MTP drafting was not observed on the long response')
        if not all(code in near.get('content', '') for code in EXPECTED):
            errors.append('Near-full early/middle/late retrieval check failed')
        results['near_full_checks'] = {'errors': errors, 'context': 262144, 'input_tokens': NEAR_FULL}

        # Observe whether this near-full state itself fits the configured backing-cache cap.
        phase['name'] = 'post_long_retention'
        offset = log_path.stat().st_size
        flush, _ = fixture('9ba0f714c5d2_post_long', 256)
        request('post_long', flush, False)
        with log_path.open('rb') as handle:
            handle.seek(offset)
            retention_log = handle.read().decode(errors='replace')
        results['near_full_state_exceeded_cache_cap'] = 'exceeds cache size limit' in retention_log
        results['cache_after_long'] = cache_snapshot(log_path.read_text(errors='replace'))
        stop.set()
        watcher.join(timeout=3)
        if guard['tripped'] or watcher.is_alive():
            raise RuntimeError(guard['reason'] or 'Memory guard did not stop cleanly')
        results['verdict'] = {'status': 'FAIL' if errors else 'PASS', 'failures': errors,
            'scope': 'Saturated 8 GiB backing cache co-resident with one 253952-token text inference; not a general quality certification',
            'near_full_state_exceeded_cache_cap': results['near_full_state_exceeded_cache_cap']}
        write_json(out / 'capacity-summary.json', results)
        print(json.dumps(results['verdict'], indent=2), flush=True)
    except BaseException as error:
        if guard['tripped']:
            raise RuntimeError('Capacity validation stopped: ' + guard['reason']) from error
        raise
    finally:
        stop.set()
        watcher.join(timeout=3)
