import json

from slime.utils.data import Dataset


class RecordingTokenizer:
    def __init__(self):
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        return "rendered prompt"


def _write_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"prompt": [{"role": "user", "content": "Solve it."}]}) + "\n")
    return path


def test_disable_thinking_is_forwarded_to_chat_template(tmp_path):
    tokenizer = RecordingTokenizer()

    Dataset(
        str(_write_dataset(tmp_path)),
        tokenizer=tokenizer,
        processor=None,
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=True,
        disable_thinking=True,
    )

    assert tokenizer.chat_template_kwargs["enable_thinking"] is False


def test_disable_thinking_overrides_template_kwargs_without_mutating_them(tmp_path):
    tokenizer = RecordingTokenizer()
    template_kwargs = {"enable_thinking": True, "custom": "value"}

    Dataset(
        str(_write_dataset(tmp_path)),
        tokenizer=tokenizer,
        processor=None,
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=True,
        apply_chat_template_kwargs=template_kwargs,
        disable_thinking=True,
    )

    assert tokenizer.chat_template_kwargs["enable_thinking"] is False
    assert tokenizer.chat_template_kwargs["custom"] == "value"
    assert template_kwargs == {"enable_thinking": True, "custom": "value"}
