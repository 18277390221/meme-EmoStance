"""Bounded, background GIF decode/scale with a wall-clock presentation mailbox.

Pillow/SQLite work stays on one worker; Tk objects are only made by the UI.
Late display frames may be skipped. PCM capture and audio are never discarded.
"""
from collections import OrderedDict, deque
from io import BytesIO
import threading
import time
import uuid

from PIL import Image


class CachedFrames:
    def __init__(self, blob, width, height, budget=32 * 1024 * 1024):
        self.source = Image.open(BytesIO(blob))
        self.cache_key = uuid.uuid4().hex
        if self.source.format not in ('GIF', 'PNG', 'JPEG', 'WEBP'):
            self.source.close()
            raise ValueError('桌面版支持GIF/PNG/JPEG/WebP。')
        self.blob = blob
        self.size = (max(1, width), max(1, height))
        self.cache = OrderedDict()
        self.bytes = 0
        self.budget = budget
        self.hits = self.misses = 0
        self.delays = {}
        self.count = None

    def frame(self, index):
        if self.count is not None and index >= self.count:
            index = 0
        if index in self.cache:
            self.hits += 1
            image = self.cache.pop(index)
            self.cache[index] = image
            return index, image, self.delays[index]
        try:
            self.source.seek(index)
        except EOFError:
            self.count = index
            index = 0
            self.source.seek(0)
            if index in self.cache:
                return self.frame(index)
        delay = max(20, min(int(self.source.info.get('duration', 100) or 100), 60000)) / 1000
        self.delays[index] = delay
        image = self.source.convert('RGBA')
        factor = min(self.size[0] / image.width, self.size[1] / image.height)
        target = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        if target != image.size:
            image = image.resize(target, Image.Resampling.BILINEAR)
        cost = image.width * image.height * 4
        while self.cache and self.bytes + cost > self.budget:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.width * old.height * 4
        if cost <= self.budget:
            self.cache[index] = image
            self.bytes += cost
        self.misses += 1
        return index, image, delay

    def close(self):
        self.source.close()
        # Images already handed to Tk's caller remain valid until its next draw.
        self.cache.clear()
        self.bytes = 0


class RenderEngine:
    def __init__(self, load_asset):
        self.load_asset = load_asset
        self.condition = threading.Condition()
        self.generation = 0
        self.pending = None
        self.events = deque()
        self.frames = deque(maxlen=2)
        self.started = None
        self.closed = False
        self.worker = threading.Thread(target=self._run, daemon=True, name='meme-render')
        self.worker.start()

    def request(self, event, size, reflow=False):
        with self.condition:
            self.generation += 1
            self.events.clear()
            self.frames.clear()
            self.started = None
            self.pending = (self.generation, dict(event), tuple(size), reflow)
            self.condition.notify_all()
            return self.generation

    def begin(self, generation, started):
        with self.condition:
            if generation == self.generation:
                self.started = started
                self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.generation += 1
            self.pending = None
            self.events.clear()
            self.frames.clear()
            self.condition.notify_all()

    def take(self, now=None):
        now = time.monotonic() if now is None else now
        with self.condition:
            events = list(self.events)
            self.events.clear()
            newest = None
            while self.frames and self.frames[0]['due'] <= now:
                newest = self.frames.popleft()
            if newest:
                events.append(newest)
            return events

    def close(self):
        with self.condition:
            self.closed = True
            self.generation += 1
            self.events.clear()
            self.frames.clear()
            self.condition.notify_all()
        self.worker.join(.25)

    def _current(self, generation):
        return not self.closed and generation == self.generation

    def _publish(self, generation, kind, **values):
        with self.condition:
            if self._current(generation):
                event = {'type': kind, 'generation': generation, **values}
                (self.frames if kind == 'frame' else self.events).append(event)

    def _wait_until(self, generation, deadline):
        with self.condition:
            while self._current(generation):
                delay = deadline - time.monotonic()
                if delay <= 0:
                    return True
                self.condition.wait(delay)
            return False

    def _run(self):
        cached = None
        cached_identity = None
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.closed:
                        self.condition.wait()
                    if self.closed:
                        return
                    generation, event, size, reflow = self.pending
                    self.pending = None
                try:
                    before = time.monotonic()
                    if event.get('background_id') is not None:
                        row = self.load_asset(event['meme_id'], background_id=event['background_id'])
                    else:
                        row = self.load_asset(event['meme_id'])
                    if not self._current(generation):
                        continue
                    identity = (event['meme_id'], event.get('background_id'),
                                row.get('source_path'), row.get('updated_at'))
                    if (cached is None or cached_identity != identity or
                            cached.blob != row['visual_blob'] or cached.size != size):
                        if cached:
                            cached.close()
                        cached = CachedFrames(row['visual_blob'], *size)
                        cached_identity = identity
                    index, image, delay = cached.frame(0)
                    duration = float(row['duration_seconds'])
                    self._publish(generation, 'ready', event=event, image=image, index=index,
                                  duration=duration, audio=row['audio_blob'], reflow=reflow,
                                  visual_name=row.get('visual_name'),
                                  cache_key=cached.cache_key,
                                  prepare_ms=round((time.monotonic() - before) * 1000, 2))
                    del row
                    with self.condition:
                        while self._current(generation) and self.started is None:
                            self.condition.wait()
                        if not self._current(generation):
                            continue
                        started = self.started
                    cursor, next_index = delay, 1
                    while (self._current(generation) and cursor < duration and
                           time.monotonic() < started + duration):
                        index, image, delay = cached.frame(next_index)
                        next_index = index + 1
                        due = started + cursor
                        # Never play old frames in a burst to catch up with sound.
                        if time.monotonic() < due + delay:
                            self._publish(generation, 'frame', image=image, index=index, due=due,
                                          cache_key=cached.cache_key)
                        cursor += delay
                        if not self._wait_until(generation, min(due, started + duration)):
                            break
                    if self._wait_until(generation, started + duration):
                        self._publish(generation, 'finished', cache_hits=cached.hits,
                                      cache_misses=cached.misses, cache_bytes=cached.bytes)
                except (OSError, ValueError, KeyError, EOFError) as error:
                    self._publish(generation, 'render_error', message='表情加载失败：' + str(error))
        finally:
            if cached:
                cached.close()
