"""Competition scope and original Race dispatch, without GPU or model calls."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class CompetitionTests(unittest.TestCase):
    def test_backend_defaults_and_queue_scope(self):
        from auto_eval.tasks_meta import get_all_versions, get_executor_task_groups, get_tasks_metadata
        from benchmark.tasks.tasks import get_available_task_types
        expected = set(get_available_task_types())
        self.assertEqual(get_all_versions(), ['v3.1'])
        self.assertEqual(set(sum(get_executor_task_groups().values(), [])), expected)
        self.assertEqual(set(get_tasks_metadata('v3.1')['default_selected']), expected)
        for backend in ['ray-vllm', 'vllm']:
            spec = importlib.util.spec_from_file_location('competition_arguments', ROOT / 'scripts' / backend / 'utils/arguments.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.assertEqual(module.BenchmarkConfig().seed, 42)
            self.assertEqual(module.BenchmarkConfig().version, 'v3.1')
            self.assertTrue(module.GenerationConfig().race_mode)
            self.assertFalse(module.PromptConfig().enable_thinking)

    def test_race_preserves_nothink_and_recommendation_split(self):
        from benchmark.generation_runner import GenerationRunner
        class Loader:
            benchmark_version = 'v3.1'
            def load_data(self, **kw):
                return {'0': {'prompt': 'synthetic', 'ground_truth': 'A', 'metadata': {}}}
            def load_data_with_thinking(self, **kw):
                return self.load_data()
        class Generator:
            def __init__(self): self.calls = []
            def __str__(self): return 'synthetic-model'
            def generate(self, prompts, **kw):
                self.calls.append(kw)
                n = kw['num_return_sequences']
                return {'0': [str(kw['enable_thinking'])] * n}, {'0': [0.] * n}
        for task, expected in [('challenge_common_sense', [(False, 64)]), ('challenge_recommendation_video_full', [(True, 32), (False, 32)])]:
            gen = Generator()
            runner = GenerationRunner(Loader())
            with tempfile.TemporaryDirectory() as d, patch.object(runner, 'save_generations') as save, patch('benchmark.generation_runner.console.print'):
                runner(task, 'test', d, gen, race_mode=True, enable_thinking=False, num_return_sequences=64, num_beams=64)
                self.assertEqual([(c['enable_thinking'], c['num_return_sequences']) for c in gen.calls], expected)
                self.assertEqual(len(save.call_args.kwargs['generations']['0']), 64)
                self.assertTrue(save.call_args.kwargs['race_mode'])

if __name__ == '__main__':
    unittest.main()
