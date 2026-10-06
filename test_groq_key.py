import importlib.util
import os
from pathlib import Path

module_path = Path(__file__).resolve().parent / 'ECD_Nexus_v3_MultiDisease' / 'ecd_nexus_v4' / 'inference_v3.py'
spec = importlib.util.spec_from_file_location('inference_v3', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
normalize_groq_key = mod.normalize_groq_key


def test_normalize_groq_key_strips_whitespace_and_env_fallback():
    os.environ.pop('GROQ_API_KEY', None)
    assert normalize_groq_key('  gsk_test_123  ') == 'gsk_test_123'

    os.environ['GROQ_API_KEY'] = '  env_key_456  '
    assert normalize_groq_key('') == 'env_key_456'
    os.environ.pop('GROQ_API_KEY', None)
