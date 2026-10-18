"""Explicit local ASR selection; no downloads or silent model fallback."""
import json
import math
from pathlib import Path


def selection(desktop=None):
    desktop = Path(desktop or Path(__file__).resolve().parent)
    config = desktop / 'asr-selection.json'
    if not config.is_file():
        # Compatibility for the previously installed lite player.
        return {'profile': 'lite', 'threads': 2, 'model': None, 'silence_rms': .008}
    data = json.loads(config.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('profile') not in ('lite', 'paraformer', 'standard'):
        raise ValueError('ASR模型配置无效，请检查desktop/asr-selection.json。')
    threads = data.get('threads', 3)
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('ASR线程数必须为1到4的整数。')
    rms = data.get('silence_rms', .008)
    if type(rms) not in (int, float) or not math.isfinite(rms) or not .0001 <= rms <= .5:
        raise ValueError('静音RMS阈值必须是0.0001到0.5的有限数值。')
    model = None
    if data['profile'] in ('paraformer', 'standard'):
        label = 'Paraformer' if data['profile'] == 'paraformer' else 'Zipformer Transducer'
        value = data.get('model')
        if not isinstance(value, str) or not value.strip():
            raise ValueError(label + '模型目录未配置，不会自动回退lite。')
        path = Path(value).expanduser()
        model = (path if path.is_absolute() else desktop / path).resolve()
        names = (('encoder.int8.onnx', 'decoder.int8.onnx', 'tokens.txt') if data['profile'] == 'paraformer'
                 else ('encoder.int8.onnx', 'decoder.onnx', 'joiner.int8.onnx', 'tokens.txt'))
        for name in names:
            if not (model / name).is_file():
                raise ValueError('本地' + label + '模型缺失：' + str(model / name))
    return {'profile': data['profile'], 'threads': threads, 'model': model, 'silence_rms': rms}
