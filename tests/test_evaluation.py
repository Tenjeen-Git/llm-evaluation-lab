import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from eval_lab.core import (compare_experiments, dataset_hash, extract_option, load_dataset,
                           model_inputs, score_answer, summary, validate_dataset, wilson_interval)
from eval_lab.runner import run_experiment
from eval_lab.langsmith_sync import publish_experiment
from eval_lab.storage import human_agreement
from eval_lab.reports import html_report, csv_export
from eval_lab.code_sandbox import evaluate_code


def test_dataset_and_reference_leak_prevention():
    tasks = load_dataset()
    assert len(tasks) == 25
    assert len({t['category'] for t in tasks}) == 5
    for task in tasks:
        assert set(model_inputs(task)) == {'question'}
        if 'options' not in task:
            assert task['reference'] not in model_inputs(task)['question']
    changed = copy.deepcopy(tasks)
    changed[0]['rubric'] += ' changed'
    assert dataset_hash(changed) != dataset_hash(tasks)


@pytest.mark.parametrize('bad', [[], {}, [1], [{'id': 'x'}]])
def test_reject_invalid_datasets(bad):
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_duplicate_ids_rejected():
    task = load_dataset()[0]
    with pytest.raises(ValueError, match='Duplicate'):
        validate_dataset([task, task])


def test_scoring_does_not_confuse_format_with_knowledge():
    task = load_dataset()[0]
    assert score_answer(task, 'Answer: B')['correctness'] == 1
    assert score_answer(task, 'Answer: B')['format_compliance'] == 0
    assert score_answer(task, 'B')['format_compliance'] == 1
    assert extract_option('A long discussion with no final option') is None
    assert extract_option('Option: A. Final answer: B') == 'B'
    assert score_answer(load_dataset()[10], 'a plausible answer')['correctness'] is None


def test_demo_results_and_paired_regressions():
    tasks = load_dataset()
    baseline = run_experiment(tasks)
    careful = run_experiment(tasks, prompt_version='careful-v2')
    assert baseline['mode'] == 'scripted_demo'
    assert summary(baseline['rows'])['mcq_accuracy'] == 0.6
    assert summary(careful['rows'])['mcq_accuracy'] == 0.8
    assert summary(careful['rows'])['judge_count'] == 0
    comparison = compare_experiments(baseline, careful)
    assert comparison['improved'] == 3
    assert comparison['regressed'] == 1
    changed = copy.deepcopy(careful)
    changed['mode'] = 'live_model'
    with pytest.raises(ValueError):
        compare_experiments(baseline, changed)


def test_demo_cannot_score_modified_tasks_or_publish():
    tasks = load_dataset()
    experiment = run_experiment(tasks)
    with pytest.raises(ValueError, match='Scripted'):
        publish_experiment(experiment, tasks)
    tasks[0]['question'] = 'A different question'
    with pytest.raises(ValueError, match='unchanged'):
        run_experiment(tasks)


def test_live_pipeline_with_fake_model_and_usage():
    captured = []
    def respond(prompt):
        captured.append(prompt.to_messages())
        return AIMessage(content='B', usage_metadata={'input_tokens': 10, 'output_tokens': 1, 'total_tokens': 11})
    experiment = run_experiment(load_dataset()[:1], provider='OpenAI', model_name='fake-test-model', model=RunnableLambda(respond))
    assert experiment['mode'] == 'live_model'
    assert experiment['rows'][0]['correctness'] == 1
    assert summary(experiment['rows'])['tokens'] == 11
    assert len(captured) == 1
    assert 'Select the single correct option' not in str(captured)


def test_failed_generation_stays_in_denominator_and_redacts_errors():
    def fail(_):
        raise RuntimeError('secret-key-must-not-be-saved')
    experiment = run_experiment(load_dataset()[:1], provider='OpenAI', model=RunnableLambda(fail))
    assert summary(experiment['rows'])['mcq_accuracy'] == 0
    assert summary(experiment['rows'])['errors'] == 1
    assert 'secret-key' not in json.dumps(experiment)


def test_agreement_only_uses_reviewed_ai_judgments():
    experiment = {'rows': [{'task_id': 'x', 'judge_score': 1}, {'task_id': 'y', 'judge_score': 0}, {'task_id': 'z', 'judge_score': None}]}
    reviews = {'x': {'score': 1}, 'y': {'score': 0}, 'z': {'score': 1}}
    assert human_agreement(experiment, reviews) == {'count': 2, 'agreement': 1.0, 'kappa': 1.0}
    assert human_agreement(experiment, {})['kappa'] is None


def test_reports_escape_model_content():
    experiment = run_experiment(load_dataset()[:1])
    experiment['rows'][0]['answer'] = '<script>alert(1)</script>'
    report = html_report(experiment)
    assert '<script>' not in report
    assert '&lt;script&gt;' in report
    assert "'=SUM" in csv_export([{'answer': '=SUM(A1:A2)'}])


def test_wilson_interval():
    low, high = wilson_interval(8, 10)
    assert 0 < low < 0.8 < high < 1
    assert wilson_interval(0, 0) is None


def test_sandbox_flags_and_cleanup_without_docker():
    task = next(t for t in load_dataset() if t['id'] == 'C01')
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({'passed': 3, 'total': 3}))
    with patch('eval_lab.code_sandbox.subprocess.run', side_effect=fake_run):
        assert evaluate_code(task, task['reference'])['code_score'] == 1
    assert '--read-only' in calls[0] and '--cap-drop' in calls[0]
    assert calls[0][calls[0].index('--network')+1] == 'none'
    assert '-v' not in calls[0] and '--mount' not in calls[0]
    assert calls[-1][1:3] == ['rm', '-f']


def test_langsmith_sync_uses_recorded_outputs_and_correct_evaluators():
    tasks = load_dataset()[:2]
    model = RunnableLambda(lambda _: AIMessage(content='B'))
    experiment = run_experiment(tasks, provider='OpenAI', model_name='fake', model=model)
    class FakeClient:
        def __init__(self):
            self.examples = []
        def create_dataset(self, **kwargs):
            return SimpleNamespace(id='dataset-id')
        def list_examples(self, **kwargs):
            return list(self.examples)
        def create_examples(self, dataset_id, examples):
            self.examples = [SimpleNamespace(**e) for e in examples]
        def evaluate(self, target, data, evaluators, **kwargs):
            for example in data:
                outputs = target(example.inputs)
                result = evaluators[0](outputs, example.outputs)
                assert 'mcq_correctness' in [r['key'] for r in result['results']]
            return SimpleNamespace(experiment_name='fake-upload')
    assert publish_experiment(experiment, tasks, FakeClient()).experiment_name == 'fake-upload'
