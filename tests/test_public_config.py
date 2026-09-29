"""Offline checks for configuration cleanup and unchanged scoring interfaces."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class PublicConfigTests(unittest.TestCase):
    def test_api_import_without_provider_sdks(self):
        code = '''import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'openai','anthropic','vertexai','google'}:
   raise ImportError('SDK deliberately unavailable')
sys.meta_path.insert(0, Block())
import api
assert callable(api.get_client)
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-c', code], env=env, check=True)

    def test_external_config_precedence_and_expansion(self):
        from api import load_config
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.json';b=Path(d)/'b.json'
            a.write_text('{"provider":{"api_key":"${TEST_CONFIG_KEY}"}}')
            b.write_text('{"chosen":"explicit"}')
            with patch.dict(os.environ, {'LLM_CONFIG_PATH':str(a),'TEST_CONFIG_KEY':'synthetic-only'}):
                self.assertEqual(load_config()['provider']['api_key'],'synthetic-only')
                self.assertEqual(load_config(str(b))['chosen'],'explicit')
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):load_config(str(a))

    def test_repr_does_not_expose_config_values(self):
        from api.base import BaseLLMClient
        class Dummy(BaseLLMClient):
            def _setup(self):pass
            def _call_api(self,*a,**kw):return ''
        client=Dummy(api_key='synthetic-secret',default_headers={'Authorization':'synthetic-header'})
        self.assertNotIn('synthetic-secret',repr(client))
        self.assertNotIn('synthetic-header',repr(client))

    def test_judge_requires_explicit_configuration(self):
        from benchmark.tasks.v3_1.challenge_sid2caption_evaluator import ChallengeSid2CaptionEvaluator
        evaluator=ChallengeSid2CaptionEvaluator({})
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):evaluator._get_llm_client()

    def test_judge_normalizes_endpoint_and_preserves_api_client(self):
        from benchmark.tasks.v3_1 import challenge_sid2caption_evaluator as module
        with patch.dict(os.environ, {'JUDGE_API_KEY':'synthetic','JUDGE_BASE_URL':'https://judge.example.org/v1/chat/completions','JUDGE_MODEL':'test-model'},clear=True):
            with patch.object(module,'_ChallengeOpenAIClient') as client:
                module.ChallengeSid2CaptionEvaluator({})._get_llm_client()
                self.assertEqual(client.call_args.kwargs['base_url'],'https://judge.example.org/v1')
                self.assertEqual(client.call_args.kwargs['model_name'],'test-model')
        self.assertEqual(module._strip_json_code_fence('```json\n{"ok":true}\n```'),'{"ok":true}')

    def test_data_alias_and_all_competition_tasks(self):
        from benchmark.tasks.tasks import get_task_config, get_available_task_types, get_evaluator
        tasks=['challenge_common_sense','challenge_evolution_action_select','challenge_evolution_topic_gen',
               *['challenge_itemic_pattern_caption_'+d for d in ['ad','product','video']],
               *['challenge_recommendation_'+d for d in ['ad','live','product','video_full']]]
        available=get_available_task_types('v3.1')
        for task in tasks:
            self.assertIn(task,available)
            self.assertTrue(callable(get_evaluator(task,'v3.1')))
        self.assertEqual(set(tasks), set(available))
        from benchmark.tasks.tasks import get_available_benchmark_versions, check_task_types
        self.assertEqual(get_available_benchmark_versions(), ['v3.1'])
        self.assertEqual(set(check_task_types(None)), set(tasks))
        with self.assertRaises(ValueError):get_task_config('challenge_recommendation_video')
        with self.assertRaises(ValueError):get_task_config(tasks[0], 'v2.0')

    def test_choice_scoring(self):
        from benchmark.tasks.tasks import get_evaluator,get_task_config
        task='challenge_common_sense'
        metrics,_=get_evaluator(task,'v3.1')({'0':{'ground_truth':'B','generations':['正确答案是 B']},'1':{'ground_truth':'A','generations':[]}},task_config=get_task_config(task,'v3.1'),overwrite=True).evaluate()
        self.assertEqual(metrics['accuracy'],0.5)

    def test_local_mapping_scoring_and_missing_mapping(self):
        from benchmark.tasks.tasks import get_evaluator,get_task_config
        task='challenge_recommendation_video_full';cfg=get_task_config(task,'v3.1');cls=get_evaluator(task,'v3.1')
        samples={'0':{'ground_truth':'<s_a_1><s_b_2><s_c_3>','metadata':{'answer_pid':[200]},'generations':['<s_a_1><s_b_2><s_c_3>']}}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):cls(samples,task_config=cfg,data_dir=d,overwrite=True).evaluate()
            (Path(d)/'sid2pid_video_ad.json').write_text(json.dumps({'67125251':[{'pid':100,'count':1},{'pid':200,'count':2}]}))
            metrics,_=cls(samples,task_config=cfg,data_dir=d,overwrite=True).evaluate()
            self.assertEqual(metrics['pass@1'],1)
            self.assertEqual(metrics['pid_pass@1'],1)

    def test_notification_hooks_do_not_send(self):
        from benchmark.lineage.reporter import report_evaluation
        from auto_eval.utils import send_message
        with patch('requests.post',side_effect=AssertionError('Unexpected network request')):
            report_evaluation({'task_id':'example'})
            send_message('unused','example')

if __name__=='__main__':unittest.main()
