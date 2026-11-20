"""Exercise the installed FormLite APK on a disposable Android emulator."""

import argparse
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def normalized(text):
    return text.lstrip('‹ ').strip().casefold().replace('’', "'")


class Emulator:
    def __init__(self, adb, serial, output):
        if not re.fullmatch(r'emulator-\d+', serial):
            raise ValueError('Use a disposable Android emulator.')
        self.prefix = [adb, '-s', serial]
        self.output = output

    def cmd(self, *arguments, binary=False):
        result = subprocess.run(self.prefix + list(arguments), check=True, capture_output=True, timeout=30)
        return result.stdout if binary else result.stdout.decode()

    def ui(self):
        self.cmd('shell', 'rm', '-f', '/sdcard/formlite-ui.xml')
        dump = self.cmd('shell', 'uiautomator', 'dump', '/sdcard/formlite-ui.xml')
        if 'dumped to' not in dump:
            return []
        xml = self.cmd('exec-out', 'cat', '/sdcard/formlite-ui.xml')
        return list(ET.fromstring(xml).iter('node'))

    def wait(self, text, timeout=24):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for node in self.ui():
                if normalized(text) in [normalized(node.get('text', '')),
                                        *map(normalized, node.get('content-desc', '').split(', '))]:
                    return node
            time.sleep(.5)
        raise AssertionError(f'Screen did not show {text!r}')

    def tap(self, text):
        node = self.wait(text)
        x1, y1, x2, y2 = map(int, re.findall(r'\d+', node.get('bounds', '')))
        self.cmd('shell', 'input', 'tap', str((x1+x2)//2), str((y1+y2)//2))

    def active_marker(self):
        events = self.cmd('logcat', '-d', '-s', 'FormLite:D', '*:S')
        return next((event for event in reversed(events.splitlines())
                     if event.endswith('Starting active session time')), None)

    def wait_for_active(self, before):
        deadline = time.monotonic() + 24
        while time.monotonic() < deadline:
            current = self.active_marker()
            if current and current != before:
                time.sleep(.8)
                return
            time.sleep(.4)
        raise AssertionError('No new active session event')

    def active_control(self, label):
        # The live timer keeps UI Automator from reaching an idle accessibility
        # state. The test fixes the viewport at 390×844 before using these bounds.
        self.cmd('shell', 'input', 'tap', '107' if label == 'Pause' else '283', '710')

    def screenshot(self, name):
        (self.output / f'{name}.png').write_bytes(self.cmd('exec-out', 'screencap', '-p', binary=True))

    def launch(self):
        self.cmd('shell', 'am', 'start', '-W', '-n', 'com.formlite/.MainActivity')
        self.wait('FormLite')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', required=True)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--output', type=Path, default=Path('verification'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    emulator = Emulator(args.adb, args.serial, args.output)
    run = emulator.cmd
    run('shell', 'pm', 'clear', 'com.formlite')
    run('shell', 'cmd', 'connectivity', 'airplane-mode', 'enable')
    run('shell', 'wm', 'size', '390x844')
    run('shell', 'wm', 'density', '160')
    try:
        emulator.launch()
        emulator.screenshot('home-reference')
        emulator.tap('Exercise guide')
        emulator.wait('Reverse crunches')
        emulator.screenshot('guide-reference')
        emulator.tap('Set up camera')
        emulator.tap("Don't allow")
        emulator.wait('Camera access is needed to recognize exercises.')
        emulator.screenshot('permission-denied')
        emulator.tap('Retry camera access')
        if any(normalized(n.get('text', '')) == 'continue' for n in emulator.ui()):
            emulator.tap('Continue')
        emulator.tap('While using the app')
        emulator.wait('Ready when you are.', timeout=45)
        emulator.screenshot('camera-setup')
        emulator.tap('Switch camera')
        emulator.wait('Ready when you are.')
        last_active = emulator.active_marker()
        emulator.tap('Start session')
        time.sleep(.45)
        emulator.screenshot('countdown')
        emulator.wait_for_active(last_active)
        emulator.screenshot('recognizing-no-pose')
        emulator.active_control('Pause')
        emulator.wait('Session paused')
        first = [n.get('text') for n in emulator.ui() if re.fullmatch(r'\d\d:\d\d', n.get('text', ''))]
        second = [n.get('text') for n in emulator.ui() if re.fullmatch(r'\d\d:\d\d', n.get('text', ''))]
        assert first and first == second
        emulator.screenshot('paused')
        last_active = emulator.active_marker()
        emulator.tap('Resume')
        emulator.wait_for_active(last_active)
        emulator.active_control('Finish')
        emulator.wait('NOT SAVED YET')
        emulator.wait('Unclassified time')
        durations = [n.get('text') for n in emulator.ui() if re.fullmatch(r'\d\d:\d\d', n.get('text', ''))]
        assert len(durations) == 2 and durations[0] == durations[1]
        emulator.screenshot('summary')
        emulator.tap('Save session')
        emulator.wait('Session saved')
        emulator.screenshot('saved-summary')
        run('shell', 'am', 'force-stop', 'com.formlite')
        emulator.launch()
        emulator.tap('History')
        emulator.wait('No exercise recognized')
        emulator.screenshot('history')
        emulator.tap('No exercise recognized')
        emulator.wait('Unclassified time')
        emulator.tap('Delete session')
        emulator.tap('Delete')
        emulator.wait('No saved sessions yet.')
        emulator.tap('Home')
        emulator.tap('Start recognition')
        emulator.wait('Ready when you are.')
        last_active = emulator.active_marker()
        emulator.tap('Start session')
        emulator.wait_for_active(last_active)
        emulator.active_control('Finish')
        emulator.tap('Discard')
        emulator.wait('Discard this session?')
        emulator.tap('Discard')
        emulator.wait('FormLite')
        emulator.tap('History')
        emulator.wait('No saved sessions yet.')
        run('shell', 'wm', 'size', '340x720')
        emulator.tap('Home')
        emulator.wait('FormLite')
        emulator.screenshot('home-small')
        print('Emulator checks passed: offline launch, denial, setup, pause/resume, review, save, restart, history, and deletion.')
    finally:
        run('shell', 'wm', 'size', 'reset')
        run('shell', 'wm', 'density', 'reset')


if __name__ == '__main__':
    main()
