"""Exercise real Streamlit rendering and offline user interactions without API calls."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_dataset_run_comparison_and_review(tmp_path, monkeypatch):
    import eval_lab.storage as storage
    monkeypatch.setattr(storage, 'RESULTS', tmp_path)
    app = Path(__file__).resolve().parents[1] / 'app.py'
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    # On a fresh checkout use the demonstration; existing local history also works.
    demo = [button for button in at.button if button.label == 'Create two demo experiments']
    if demo:
        demo[0].click().run()
    for page in ['Dataset Studio', 'Run Experiments', 'Compare Results', 'Human Review', 'LangSmith & Guide', 'Dashboard']:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.message for e in at.exception])
    at.sidebar.radio[0].set_value('Run Experiments').run()
    next(button for button in at.button if button.label == 'Run both experiments').click().run()
    assert not at.exception
    assert len(storage.history()) == 4
    at.sidebar.radio[0].set_value('Human Review').run()
    next(box for box in at.selectbox if box.label == 'Task to review').set_value('T01').run()
    next(radio for radio in at.radio if radio.label == 'Your assessment').set_value('Correct')
    next(area for area in at.text_area if area.label == 'Reason or correction').set_value('Reviewed independently in the smoke test.')
    next(button for button in at.button if button.label == 'Save review').click().run()
    assert not at.exception
    reviewed = [storage.load_reviews(e['id']) for e in storage.history()]
    assert any(reviews.get('T01', {}).get('score') == 1 for reviews in reviewed)
