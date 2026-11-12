"""The prompt that a Clef checkpoint's joint schema head reads.

Clef answers /v1/systemone questions from the backbone's hidden states instead
of from generated text. One prompt holds the state and every question with its
allowed options, and the head scores the span of each option. This follows
encode_record in the checkpoint's joint_schema_model.py, piece by piece, since
each piece is tokenized on its own.
"""

from __future__ import annotations

import functools
import json
from typing import Any, List, Sequence, Tuple

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRequest,
)
from sglang.srt.layers.joint_schema_head import (
    QUESTION_TYPES,
    LayoutQuestion,
    pack_decision_layout,
)

try:  # transformers v5
    from transformers.tokenization_utils_tokenizers import (
        TokenizersBackend as _TokenizersBackend,
    )
except ImportError:
    _TokenizersBackend = None
_OVERRIDES = frozenset(
    (
        "encode",
        "_get_padding_truncation_strategies",
        "_encode_plus",
        "set_truncation_and_padding",
        "_convert_encoding",
    )
)

_SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
_SYSTEM = f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
_PREFIX = f"{_SYSTEM}STATE:\n"
# GreenLeaf "sandwich": the fields come once before the state and again after
# it; the head reads the spans of the second copy, which has seen the state.
_SANDWICH_PREFIX = f"{_SYSTEM}SCHEMA FIELDS:\n"
_SANDWICH_STATE = "\nSTATE:\n"
LAYOUTS = ("state_first", "sandwich")
_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
_SUFFIX = (
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "JOINT SCHEMA DECISIONS:"
)
_NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
# The default max_length of encode_record and systemone in joint_schema_model.py.
REFERENCE_MAX_LENGTH = 16384


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _state_ids(tokenizer: Any, text: str) -> List[int]:
    """``tokenizer.encode(text, add_special_tokens=False)``, by one call to the
    Rust backend when that call gives the same ids.

    The stock HF v5 fast encode (no padding, no truncation) clears the
    backend's truncation and padding, sets its ``encode_special_tokens`` to
    ``split_special_tokens`` and returns
    ``backend.encode_batch([text], add_special_tokens=False)[0].ids``. With the
    backend already in that state and ``split_special_tokens`` off,
    ``encode_batch_fast`` runs the same pipeline without computing character
    offsets, which the ids do not use. Any other tokenizer or backend state
    takes the original encode.
    """
    backend = getattr(tokenizer, "_tokenizer", None)
    cls = type(tokenizer)
    if (
        _TokenizersBackend is not None
        and type(backend) is Tokenizer
        and getattr(tokenizer, "is_fast", False)
        # Only the stock path: encode -> _encode_plus -> backend.encode_batch.
        and cls.encode is PreTrainedTokenizerBase.encode
        and cls._get_padding_truncation_strategies
        is PreTrainedTokenizerBase._get_padding_truncation_strategies
        and cls._encode_plus is _TokenizersBackend._encode_plus
        and cls.set_truncation_and_padding
        is _TokenizersBackend.set_truncation_and_padding
        and cls._convert_encoding is _TokenizersBackend._convert_encoding
        and not _OVERRIDES.intersection(vars(tokenizer))
        and not getattr(tokenizer, "split_special_tokens", False)
        # The state that encode would set before calling the backend.
        and backend.truncation is None
        and backend.padding is None
        and not backend.encode_special_tokens
    ):
        return backend.encode_batch_fast([text], add_special_tokens=False)[0].ids
    return tokenizer.encode(text, add_special_tokens=False)


@functools.lru_cache(maxsize=4096)
def _piece_ids(tokenizer: Any, text: str) -> Tuple[int, ...]:
    """The ids of a prompt piece other than the state. Requests with the same
    questions repeat these pieces, and tokenizing each one again costs ~2 ms
    a request, more than the state."""
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def joint_schema_options(question: SystemOneQuestion) -> List[Tuple[str, Any]]:
    """Each option's name and description, in the order the head scores them."""
    if isinstance(question, SystemOneNoulQuestion):
        criteria = dict(_NOUL_CRITERIA)
        if question.criteria is not None:
            # A description sent as null drops the default, as in encode_record.
            criteria.update(question.criteria.model_dump(exclude_unset=True))
        return [(name, criteria[name]) for name in ("true", "false")]
    if isinstance(question, SystemOneChoiceQuestion):
        return sorted(question.criteria.items(), key=lambda item: item[0])
    return [(str(level), detail) for level, detail in enumerate(question.criteria)]


def encode_joint_schema(
    tokenizer: Any,
    request: SystemOneRequest,
    max_length: int,
    image_token_counts: Sequence[int],
    layout: str = "state_first",
) -> Tuple[List[int], List[int]]:
    """The prompt ids and their decision layout.

    Each image's placeholder expands to its count in ``image_token_counts``, and
    the state is truncated so that the expanded prompt has at most
    ``max_length`` tokens, as encode_record budgets the expanded media.
    """

    def tokens(text: str) -> List[int]:
        return list(_piece_ids(tokenizer, text))

    if layout not in LAYOUTS:
        raise ValueError(f"unknown joint schema layout {layout!r}")
    if layout == "sandwich" and request.images:
        raise ValueError("a sandwich-layout checkpoint takes no images")
    header = tokens("\n\nSCHEMA FIELDS:\n")
    schema = []
    questions = []
    for index, (question_id, question) in enumerate(request.questions.items()):
        schema += tokens(
            f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {question.type}\n"
            "INSTRUCTION: "
        )
        instructions = question.instructions
        if instructions is None or instructions == "":
            instructions = question_id
        question_start = len(schema)
        schema += tokens(_render(instructions))
        question_span = (question_start, len(schema))
        schema += tokens("\nALLOWED OPTIONS:\n")
        option_spans = []
        for option_index, (name, description) in enumerate(
            joint_schema_options(question)
        ):
            schema += tokens(f"OPTION {option_index + 1}: ")
            option_start = len(schema)
            semantics = {"option_id": name}
            if description is not None:
                semantics["description"] = description
            schema += tokens(_render(semantics))
            option_spans.append((option_start, len(schema)))
            schema += tokens("\n")
        schema += tokens("END FIELD\n")
        questions.append(
            (QUESTION_TYPES.index(question.type), question_span, option_spans)
        )

    if layout == "sandwich":
        prefix = tokens(_SANDWICH_PREFIX) + schema + tokens(_SANDWICH_STATE)
    else:
        prefix = tokens(_PREFIX)
    if request.images:
        prefix += tokens(_IMAGE * len(request.images) + "\n")
    suffix = tokens(_SUFFIX)
    # Each image's single <|image_pad|> becomes that image's tokens.
    expansion = sum(count - 1 for count in image_token_counts)
    fixed_length = len(prefix) + expansion + len(header) + len(schema) + len(suffix)
    if fixed_length > max_length:
        raise ValueError(
            f"the questions need {fixed_length} prompt tokens before the state, "
            f"but the prompt holds at most {max_length}"
        )
    state = _state_ids(tokenizer, _render(request.state))
    budget = max_length - fixed_length
    if layout == "sandwich" and len(state) > budget:
        # A guard must read the newest message, at the end of the state:
        # keep the last tokens.
        state = state[len(state) - budget :]
    else:
        state = state[:budget]
    offset = len(prefix) + len(state) + len(header)
    input_ids = prefix + state + header + schema + suffix
    layout = pack_decision_layout(
        len(input_ids),
        [
            LayoutQuestion(
                question_type,
                (question_span[0] + offset, question_span[1] + offset),
                tuple((start + offset, end + offset) for start, end in option_spans),
            )
            for question_type, question_span, option_spans in questions
        ],
    )
    return input_ids, layout
