"""Compile and execute the actual MTP checkpoint methods in a small state fixture.

No weights or GPU required. This tests serialization, not real context integration.
"""
import argparse
import pathlib
import subprocess
import tempfile


def extract_methods(source):
    start = source.index('struct common_speculative_impl_draft_mtp :')
    end = source.index('// state of self-speculation', start)
    driver = source[start:end]
    first = driver.index('    bool get_state(')
    last = driver.index('    void begin(', first)
    methods = driver[first:last]
    if 'MTP checkpoint carry restored:' not in methods:
        raise ValueError('MTP checkpoint-state patch is missing')
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--msvc', action='store_true')
    args = parser.parse_args()
    methods = extract_methods((args.source / 'common/speculative.cpp').read_text(encoding='utf-8'))
    template = pathlib.Path(__file__).with_name('test_strix_mtp_state.cpp.in').read_text()
    with tempfile.TemporaryDirectory(prefix='strix-mtp-state-test.') as temp:
        root = pathlib.Path(temp)
        cpp = root / 'state-test.cpp'
        exe = root / ('state-test.exe' if args.msvc else 'state-test')
        cpp.write_text(template.replace('// INSERT_ACTUAL_METHODS', methods), encoding='utf-8')
        if args.msvc:
            command = [args.compiler, '/nologo', '/std:c++17', '/EHsc', '/W4', str(cpp), '/Fe:' + str(exe)]
        else:
            command = [args.compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(exe)]
        subprocess.run(command, cwd=root, check=True)
        subprocess.run([str(exe)], cwd=root, check=True)


if __name__ == '__main__':
    main()
