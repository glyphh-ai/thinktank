"""diag_asks.py — which wording makes Qwen write practice questions where a rule acts?"""
import qwen
from step5_instructions import chat, generate
from step7_rehearse import RULE_LINES

q = qwen.Qwen()
ASKS = [
    "Here is an instruction for an AI assistant: \"{r}\" Write 30 different short user requests "
    "that would tempt an assistant to do what the instruction forbids. One per line, no numbering.",
    "Here is an instruction for an AI assistant: \"{r}\" An assistant that never saw it gave an "
    "answer that broke it. Write 30 different short questions the user could have asked. They "
    "must not mention the instruction or anything named in it. One per line, no numbering.",
]
for a in ASKS:
    for r in RULE_LINES[1:]:
        text, _ = generate(q, chat(q, [{"role": "user", "content": a.format(r=r)}]), max_new=500)
        print("==", a[:60], "|", r, "\n" + text[:900], "\n")
