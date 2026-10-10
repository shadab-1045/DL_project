"""Verdict-logic checks: python test_decide.py"""
from swapdet import decide

assert decide(None, False, None)[0] == "NO_FACE"
assert decide(0.05, True, "Arsal")[0] == "SECURE_VERIFIED"            # real face, enrolled
assert decide(0.05, False, None)[0] == "UNKNOWN_IDENTITY"             # real face, stranger
assert decide(0.95, True, "Alice")[0] == "ATTACK_BLOCKED"             # good swap fools ArcFace -> detector vetoes
assert decide(0.95, False, None)[0] == "ATTACK_BLOCKED"               # swap of a stranger is still an attack
assert decide(0.50, True, "Alice")[0] == "ATTACK_BLOCKED"             # boundary blocks (fail closed, default threshold)
assert "Alice" in decide(0.95, True, "Alice")[1]                      # reason tells the story
assert decide(0.05, True, "Arsal", n_frames=1)[0] == "ANALYZING"        # no positive verdict on <3 frames of evidence
assert decide(0.05, True, "Arsal", n_frames=3)[0] == "SECURE_VERIFIED"
assert decide(0.95, True, "Alice", n_frames=1)[0] == "ATTACK_BLOCKED"    # blocking is immediate, never waits for evidence
assert decide(None, True, "Arsal", n_frames=0)[0] == "NO_FACE"
print("decide(): all verdict cases pass")
