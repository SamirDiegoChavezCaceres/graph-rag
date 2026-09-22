"""Turn text into (subject, relation, object) triples.

The default extractor is rule-based and dependency-free: it finds proper-noun
entities and a known relation phrase between them, so the demo and tests run
offline and deterministically. Real, messy text needs an LLM extractor, which
plugs in behind the same ``Extractor`` interface (the ``openai`` extra).
"""

from __future__ import annotations

import re
from typing import List, Protocol, Tuple

Triple = Tuple[str, str, str]

# Relation phrases we recognize, longest first so "based in" wins over "in".
RELATIONS = [
    "studied at", "works at", "based in", "located in", "part of",
    "acquired by", "founded", "acquired", "built", "develops", "created",
    "owns", "uses", "wrote", "leads",
]

_ENTITY = re.compile(r"[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*")
_SENT = re.compile(r"[^.!?]+")


class Extractor(Protocol):
    def extract(self, text: str) -> List[Triple]: ...


class RuleBasedExtractor:
    def __init__(self, relations: List[str] = None) -> None:
        self.relations = relations or RELATIONS

    def extract(self, text: str) -> List[Triple]:
        triples: List[Triple] = []
        for sent in _SENT.findall(text):
            spans = [(m.group(0).strip(), m.start(), m.end())
                     for m in _ENTITY.finditer(sent)]
            # look at each adjacent entity pair for a relation phrase between them
            for (e1, _s1, end1), (e2, start2, _e2) in zip(spans, spans[1:]):
                between = sent[end1:start2].lower()
                rel = next((r for r in self.relations if r in between), None)
                if rel:
                    triples.append((e1, rel.replace(" ", "_"), e2))
        return triples


class LLMExtractor:
    """Extract triples with an OpenAI model (the ``openai`` extra).

    Reads ``OPENAI_API_KEY`` from the environment or a local ``.env`` file and
    asks for JSON triples. Same interface as the rule-based one.
    """

    def __init__(self, model: str = None) -> None:
        import os

        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except Exception:
            pass
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def extract(self, text: str) -> List[Triple]:
        import json

        prompt = (
            "Extract factual relationships as JSON: "
            '{"triples": [["subject", "relation", "object"], ...]}. '
            "Use short relation labels. Text:\n\n" + text
        )
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return [tuple(t) for t in data.get("triples", []) if len(t) == 3]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []


def extract_triples(text: str, extractor: Extractor = None) -> List[Triple]:
    return (extractor or RuleBasedExtractor()).extract(text)
