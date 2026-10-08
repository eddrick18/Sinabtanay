"""Optional Windows speech in a hidden child process; webcam loop never waits for speech."""
import base64
import os
from pathlib import Path
import subprocess


class LocalSpeech:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.process = None
        self.status = 'T: speak | Esc: stop' if enabled else 'Speech disabled'
        self.executable = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        self.helper = Path(__file__).with_name('speak.ps1')

    def poll(self):
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.process = None
                self.status = 'Speech finished' if code == 0 else 'Speech unavailable - check Windows voice/audio'
        return self.status

    def speak(self, text):
        self.poll()
        if not self.enabled:
            return self.status
        if self.process is not None:
            return 'Already speaking - Esc: stop'
        text = text.strip()
        if not text:
            return 'Message empty - add letters first'
        if len(text) > 200:
            return 'Message too long for speech'
        if os.name != 'nt' or not self.executable.is_file():
            self.status = 'Local speech requires Windows PowerShell'
            return self.status
        try:
            self.process = subprocess.Popen(
                [str(self.executable), '-NoProfile', '-NonInteractive', '-File', str(self.helper)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            # At most 200 characters: bounded pipe write, with no command interpolation.
            self.process.stdin.write(base64.b64encode(text.encode('utf-8')))
            self.process.stdin.close()
            self.status = 'Speaking message | Esc: stop'
        except OSError:
            self.stop()
            self.status = 'Could not start local speech'
        return self.status

    def stop(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
            except OSError:
                pass
            self.process = None
        self.status = 'Speech stopped' if self.enabled else 'Speech disabled'
        return self.status

    def close(self):
        self.stop()
