"""Regression coverage for the results template; no checkpoint download/training."""

from html.parser import HTMLParser
import unittest

from app import app, RESULTS_CACHE


class ResultCards(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.current = None
        self.cards = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            if attrs.get("class") == "result-card":
                self.current = []
                self.depth = 1
            elif self.current is not None:
                self.depth += 1

    def handle_endtag(self, tag):
        if tag == "div" and self.current is not None:
            self.depth -= 1
            if self.depth == 0:
                self.cards.append(" ".join(self.current))
                self.current = None

    def handle_data(self, data):
        if self.current is not None and data.strip():
            self.current.append(data.strip())


class ResultsPageTest(unittest.TestCase):
    def setUp(self):
        self.session_id = "documentation-regression"
        images = []
        for actual in ("NORMAL", "PNEUMONIA"):
            for correct in (True, False):
                predicted = actual if correct else ("PNEUMONIA" if actual == "NORMAL" else "NORMAL")
                filename = f"{actual.lower()}_{'correct' if correct else 'incorrect'}.jpeg"
                images.append({"filename": filename, "actual": actual,
                               "predicted": predicted, "correct": correct,
                               "confidence": 0.75, "path": f"dataset/{filename}"})
        RESULTS_CACHE[self.session_id] = {
            "normal_results": images[:2], "pneumonia_results": images[2:],
            "accuracy": 0.5, "sensitivity": 0.5, "specificity": 0.5,
            "precision": 0.5, "f1_score": 0.5,
            "confusion_matrix": {"true_positive": 1, "true_negative": 1,
                                 "false_positive": 1, "false_negative": 1},
            "training_info": {"total_training_images": 8, "normal_count": 4, "pneumonia_count": 4},
        }
        self.client = app.test_client()

    def tearDown(self):
        RESULTS_CACHE.pop(self.session_id, None)

    def test_results_render_all_five_tabs_with_consistent_badges(self):
        response = self.client.get(f"/results/{self.session_id}")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for tab in ("all", "normal", "pneumonia", "correct", "incorrect"):
            self.assertIn(f'id="{tab}-results"', html)
        parser = ResultCards()
        parser.feed(html)
        # 4 all + 2 normal + 2 pneumonia + 2 correct + 2 incorrect.
        self.assertEqual(len(parser.cards), 12)
        for card in parser.cards:
            self.assertEqual(card.count("Diagnosis:"), 1)
            expected = "Incorrect" if "_incorrect.jpeg" in card else "Correct"
            self.assertIn(f"Diagnosis: {expected}", card)

    def test_result_images_use_web_paths(self):
        html = self.client.get(f"/results/{self.session_id}").get_data(as_text=True)
        self.assertIn('/static/dataset/normal_correct.jpeg', html)
        self.assertNotIn('dataset\\', html)
        self.assertNotIn('%5C', html)

    def test_unknown_results_redirect_to_selection(self):
        response = self.client.get('/results/not-a-recorded-experiment')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/')


if __name__ == "__main__":
    unittest.main()
