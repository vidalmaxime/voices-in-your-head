import unittest
import xml.etree.ElementTree as ET

from prepare_notebooks import candidates, prose


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class PreparationTests(unittest.TestCase):
    def test_note_markers_do_not_join_words_or_leave_digits(self):
        p = ET.fromstring('<p>I paused<a role="doc-noteref">12</a>, then <i>continued</i>.</p>')
        self.assertEqual(prose(p), "I paused, then continued.")

    def test_targets_are_intact_sentence_tails(self):
        sentence = "I thought that I needed another answer because I had never questioned the question itself."
        found = list(candidates(sentence, WordTokenizer()))
        self.assertTrue(found)
        for _, cuts in found:
            for _, target in cuts:
                self.assertTrue(sentence.endswith(target))
                self.assertTrue(target.endswith('.'))

    def test_short_tails_are_rejected_and_bounds_are_configurable(self):
        sentence = "I thought that I needed another answer because I had never questioned the question itself."
        tails = [len(t.split()) for _, cuts in candidates(sentence, WordTokenizer()) for _, t in cuts]
        self.assertTrue(tails)
        self.assertTrue(all(8 <= n <= 24 for n in tails))
        wide = [len(t.split()) for _, cuts in candidates(sentence, WordTokenizer(), target_words=(4, 14)) for _, t in cuts]
        self.assertLess(min(wide), 8)
        self.assertGreater(len(wide), len(tails))

    def test_quoted_and_second_person_sentences_are_excluded(self):
        tokenizer = WordTokenizer()
        self.assertEqual(list(candidates('I said “you need to think about this in a different way.”', tokenizer)), [])
        self.assertEqual(list(candidates('I believe you should spend more time thinking about the question itself.', tokenizer)), [])

    def test_impersonal_and_question_sentences_are_kept(self):
        tokenizer = WordTokenizer()
        self.assertTrue(list(candidates('Humility makes the difference between art of the very first order and all the rest of art.', tokenizer)))
        self.assertTrue(list(candidates('How could there possibly be a good which was something in itself, independent of a mind that conceives it?', tokenizer)))

    def test_notation_and_unbalanced_parentheses_are_excluded(self):
        tokenizer = WordTokenizer()
        self.assertEqual(list(candidates('A truth is the unnameable (ἄλογος) point with reference to which one can order all possible opinions on a subject.', tokenizer)), [])
        self.assertEqual(list(candidates('(Has the presence in each sex of secondary characteristics of the opposite sex anything to do with this interior form of fertilization?', tokenizer)), [])
        self.assertEqual(list(candidates('The pages to reread are 120, 121, 123, 129, 173, 191, 203, 209, 241, 310, 313, 339, 373 and 415.', tokenizer)), [])
        self.assertEqual(list(candidates("It asks: ‘Why so?' One must reply: because that is; if that is, there is a reason for it.", tokenizer)), [])

    def test_dash_becomes_spoken_word_boundary(self):
        found = list(candidates('I question happiness—we need to decide what kind of life matters to us.', WordTokenizer()))
        self.assertTrue(found)
        for _, cuts in found:
            for fragment, _ in cuts:
                self.assertNotIn('happinesswe', fragment)


if __name__ == '__main__':
    unittest.main()
