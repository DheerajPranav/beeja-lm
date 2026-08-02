"""A tiny embedded corpus for offline smoke training and tests.

Real datasets are downloaded reproducibly in a later stage. Until then this
short, original, public-domain-style text gives the character models enough
structure (letters, spaces, punctuation, repetition) to measurably reduce loss
without any network access.
"""

from __future__ import annotations

SAMPLE_TEXT = """A seed is a small promise.
Given soil, water, and patience, the seed becomes a tree, and the tree becomes a forest.
Beeja means seed, and this model is a seed too.
It starts as random noise and learns, one character at a time, to guess what comes next.
First it learns the shape of words: a space follows a word, a period follows a sentence.
Then it learns that letters lean on the letters before them, that a q leans toward a u.
The model is small, the lessons are patient, and the goal is understanding, not size.
Grow slowly. Test everything. Keep the seed honest.
A seed is a small promise, and a small promise, kept, becomes a forest.
"""
