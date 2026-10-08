"""Nonblocking capture timer; each resume includes preparation time."""
import math

class CaptureTimer:
    def __init__(self, interval=0.75, countdown=3.0):
        if not math.isfinite(interval) or not math.isfinite(countdown) or interval < 0.5 or countdown < 0:
            raise ValueError('Interval must be at least 0.5 seconds and countdown nonnegative.')
        self.interval, self.countdown = interval, countdown
        self.next_at = None

    def toggle(self, now):
        self.next_at = now + self.countdown if self.next_at is None else None

    def pause(self):
        self.next_at = None

    def due(self, now):
        if self.next_at is None or now < self.next_at:
            return False
        self.next_at = now + self.interval
        return True

    def status(self, now):
        if self.next_at is None:
            return 'Paused - S: start'
        return f'Timed capture | next attempt in {max(0, self.next_at-now):.1f}s | S: pause'
