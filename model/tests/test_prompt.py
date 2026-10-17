import unittest

import torch

from emostance_meme.backbone import LocalMistral


class CharacterTokenizer:
    bos_token_id = 1
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]
    def decode(self, tokens, skip_special_tokens=True):
        return ''.join(chr(token) for token in tokens)
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # Start/end markers emulate the required Mistral instruction wrapper.
        result = [1]
        for message in messages:
            result.extend([2 if message['role'] == 'user' else 4])
            result.extend(self.encode(message['content']))
            result.append(3 if message['role'] == 'user' else 5)
        return result


class PromptTests(unittest.TestCase):
    def backbone(self, limit):
        result = LocalMistral.__new__(LocalMistral)
        result.tokenizer = CharacterTokenizer()
        result.device = torch.device('cpu')
        result.max_context_tokens = limit
        return result

    def test_long_single_sentence_keeps_complete_instruction_wrapper(self):
        ids = self.backbone(8).prompt_ids([{'role': 'user', 'content': 'abcdefghij'}]).tolist()[0]
        self.assertEqual(ids[:2], [1, 2])
        self.assertEqual(ids[-1], 3)
        self.assertEqual(ids[2:-1], [ord(char) for char in 'fghij'])

    def test_context_trimming_drops_complete_old_pairs(self):
        ids = self.backbone(8).prompt_ids([
            {'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'now'},
        ]).tolist()[0]
        self.assertEqual(ids, [1, 2] + [ord(char) for char in 'now'] + [3])

    def test_impossibly_short_context_budget_fails_clearly(self):
        with self.assertRaises(ValueError):
            self.backbone(2).prompt_ids([{'role': 'user', 'content': 'abcdefghij'}])


if __name__ == '__main__':
    unittest.main()
