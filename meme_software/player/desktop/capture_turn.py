"""Microphone windows: endpoint final is emitted only after ALSA closes."""
import threading
import time

class MicrophoneGate:
    def __init__(self, enabled=True):
        self.condition = threading.Condition()
        self.enabled, self.stopped, self.epoch = enabled, False, 0
        self.active = False
        self.pause_id = 'initial' if not enabled else None

    def command(self, name, pause_id=None):
        with self.condition:
            if name in ('stop', 'pause'):
                self.epoch += 1
                self.enabled = False
                self.stopped = self.stopped or name == 'stop'
                if pause_id is not None:
                    self.pause_id = pause_id
            elif name == 'resume' and not self.stopped:
                self.enabled = True
                self.pause_id = None
            self.condition.notify_all()
            return name == 'pause' and not self.active

    def begin(self, epoch):
        with self.condition:
            if not self.enabled or self.stopped or epoch != self.epoch:
                return False
            self.active = True
            return True

    def closed(self):
        with self.condition:
            self.active = False
            return self.pause_id

    def wait(self):
        with self.condition:
            while not self.enabled and not self.stopped:
                self.condition.wait()
            return None if self.stopped else self.epoch

    def current(self, epoch):
        with self.condition:
            return self.enabled and not self.stopped and self.epoch == epoch


def listen_once(gate, epoch, command, make_decoder, send, capture_factory):
    if not gate.begin(epoch):
        return None
    final_text, done, gated_bytes = [], threading.Event(), 0

    def finalized(text, **_):
        if gate.current(epoch) and text.strip() and not final_text:
            final_text.append(text.strip())
            gate.command('pause')

    decoder = make_decoder(finalized)
    with capture_factory(command, strict=True, frame_ms=20, new_session=False) as capture:
        def monitor():
            while not done.wait(.02):
                if not gate.current(epoch):
                    try:
                        capture.request_finish()
                    except OSError as error:
                        send('error', message='暂停麦克风失败：' + str(error),
                             reason='audio_control_error', recoverable=False)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        last_activity = last_stats = 0.0
        heard = False
        speech_started_at = None
        try:
            for block in capture.chunks():
                if not gate.current(epoch):
                    # Samples arriving after the listening window closed are
                    # explicitly counted, never passed to a later ASR stream.
                    gated_bytes += len(block)
                    continue
                if not heard:
                    heard = True
                    send('listening')
                decoder.feed(block, 16000)
                now = time.monotonic()
                if getattr(decoder, 'utterance_seen', False) and speech_started_at is None:
                    speech_started_at = now
                if speech_started_at is not None and now - speech_started_at >= 25:
                    # Do not turn a long unfinished fragment into a forced final.
                    gate.command('pause')
                if now - last_activity >= .2:
                    send('activity')
                    last_activity = now
                if now - last_stats >= 10:
                    send('stats', data=dict(decoder.stats(), capture=capture.stats()))
                    last_stats = now
            # Pause/stop must not call finish(): an unfinished phrase would
            # become a forced final despite the absence of a silence endpoint.
        finally:
            done.set()
            watcher.join(timeout=4.2)
    statistics = dict(decoder.stats(), capture=capture.stats(),
                      gated_after_turn_samples=gated_bytes // 2)
    send('stats', data=statistics)
    send('paused', pause_id=gate.closed())  # ALSA exited before any final/send.
    if final_text and not gate.stopped:
        send('final', text=final_text[0])
    elif getattr(decoder, 'utterance_seen', False) and not gate.stopped:
        send('utterance_cancelled', message='本句话未完整结束，已释放收音；请重说。')
    return statistics
