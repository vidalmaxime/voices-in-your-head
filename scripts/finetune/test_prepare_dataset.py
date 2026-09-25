import random
import unittest

from prepare_dataset import pair_from_sentence, read_system_prompt, to_messages
from pathlib import Path


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class OversizeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(32))


class PersonalDatasetTests(unittest.TestCase):
    def test_preserves_long_sentence_tail_instead_of_truncating(self):
        sentence = ('I keep returning to this idea because my quiet morning walks reveal '
                    'small details hidden beneath familiar routines throughout the changing seasons of life.')
        for seed in range(10):
            pair = pair_from_sentence(sentence, random.Random(seed), WordTokenizer())
            self.assertIsNotNone(pair)
            fragment, target = pair
            self.assertTrue(sentence.endswith(target))
            self.assertTrue(target.endswith('.'))
            self.assertGreaterEqual(len(fragment.split()), 5)
            self.assertLessEqual(len(target.split()), 24)

    def test_reserves_eos_within_32_token_generation_budget(self):
        sentence = 'I keep returning to this idea because my quiet morning walks reveal something new.'
        self.assertIsNotNone(pair_from_sentence(sentence, random.Random(7), WordTokenizer()))
        self.assertIsNone(pair_from_sentence(sentence, random.Random(7), OversizeTokenizer()))

    def test_matches_runtime_prompt_without_few_shot_examples(self):
        prompt = read_system_prompt(Path(__file__).resolve().parents[2])
        messages = to_messages(prompt, 'i keep thinking about', 'the shape of tomorrow.')['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant'])
        self.assertEqual(messages[0]['content'], prompt)
        self.assertEqual(messages[1]['content'], 'Text before cursor: i keep thinking about\nText after cursor:')


if __name__ == '__main__':
    unittest.main()
