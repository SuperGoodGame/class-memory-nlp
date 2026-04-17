import argparse
from dataclasses import dataclass

from api_utils import chat_completion, describe_chat_target
from data_utils import load_text

DATA_PATH = "data/books/alice_in_wonderland.md"


# ─── Evaluation dataset ────────────────────────────────────────────────────
@dataclass
class TestCase:
    query: str
    expected_in_answer: list[str]
    description: str


EVAL_DATASET = [
    TestCase(
        query="What unusual combination made Alice realize the Rabbit was extraordinary before she followed it?",
        expected_in_answer=["waistcoat-pocket", "watch"],
        description="Rabbit anomaly: pocket + watch",
    ),
    TestCase(
        query="What was written on the jar Alice picked up while falling?",
        expected_in_answer=["orange marmalade"],
        description="Falling scene object label",
    ),
    TestCase(
        query="Why did Alice avoid dropping the marmalade jar?",
        expected_in_answer=["killing somebody underneath", "kill somebody underneath"],
        description="Reason for keeping the jar",
    ),
    TestCase(
        query="What two grand geographical words did Alice say while falling, even though she did not understand them?",
        expected_in_answer=["latitude", "longitude"],
        description="Geography terms during fall",
    ),
    TestCase(
        query="Which two countries did Alice wonder about while imagining she might come out on the other side of the earth?",
        expected_in_answer=["new zealand", "australia"],
        description="Antipodes speculation",
    ),
    TestCase(
        query="What pair of reversed questions did Alice drowsily repeat to herself about animals in the air?",
        expected_in_answer=["do cats eat bats", "do bats eat cats"],
        description="Drowsy self-questioning",
    ),
    TestCase(
        query="What kind of table did Alice find in the hall before discovering the little key?",
        expected_in_answer=["three-legged table", "glass"],
        description="Hall furniture detail",
    ),
    TestCase(
        query="How high was the little door behind the curtain?",
        expected_in_answer=["fifteen inches", "15 inches"],
        description="Door size",
    ),
    TestCase(
        query="Before drinking from the bottle, what warning word did Alice look for on it?",
        expected_in_answer=["poison"],
        description="Bottle safety check",
    ),
    TestCase(
        query="What mixed flavor did the bottle taste like? Name at least two items.",
        expected_in_answer=["cherry-tart", "custard", "pine-apple", "roast turkey", "toffee", "hot buttered toast"],
        description="Bottle flavor list",
    ),
    TestCase(
        query="After drinking, about how tall did Alice become when she thought she could fit through the little door?",
        expected_in_answer=["ten inches", "10 inches"],
        description="Post-drink size",
    ),
    TestCase(
        query="What words were beautifully marked in currants on the cake?",
        expected_in_answer=["eat me"],
        description="Cake inscription",
    ),
    TestCase(
        query="When Alice grew enormously, what practical gift did she imagine sending to her own feet every Christmas?",
        expected_in_answer=["a new pair of boots", "boots"],
        description="Gift to her feet",
    ),
    TestCase(
        query="What incorrect multiplication answer did Alice give for four times five while doubting who she was?",
        expected_in_answer=["twelve", "12"],
        description="Faulty arithmetic",
    ),
    TestCase(
        query="Which animal did Alice first try addressing in French with 'Où est ma chatte?'",
        expected_in_answer=["mouse"],
        description="French line target",
    ),
    TestCase(
        query="What specific kind of dog did Alice describe while trying to talk to the Mouse?",
        expected_in_answer=["terrier", "bright-eyed terrier"],
        description="Dog description",
    ),
    TestCase(
        query="Who proposed the Caucus-race as a way to get everyone dry?",
        expected_in_answer=["dodo"],
        description="Caucus-race proposer",
    ),
    TestCase(
        query="According to the Dodo, who won the Caucus-race?",
        expected_in_answer=["everybody", "all must have prizes"],
        description="Race result",
    ),
    TestCase(
        query="What item from Alice's pocket was ceremonially presented back to her as a prize?",
        expected_in_answer=["thimble"],
        description="Prize irony",
    ),
    TestCase(
        query="What did Alice say the Mouse's sad tale looked like to her?",
        expected_in_answer=["long tail", "tail"],
        description="Tale/tail pun",
    ),
    TestCase(
        query="What mistaken name did the White Rabbit call Alice when sending her to fetch gloves and a fan?",
        expected_in_answer=["mary ann"],
        description="Mistaken identity",
    ),
    TestCase(
        query="What name was engraved on the brass plate at the Rabbit's house?",
        expected_in_answer=["w rabbit", "w. rabbit"],
        description="House nameplate",
    ),
    TestCase(
        query="What did Pat say was in the window when the Rabbit asked him?",
        expected_in_answer=["an arm", "arm"],
        description="Window confusion",
    ),
    TestCase(
        query="What did the Rabbit's helpers throw through the window that later changed form?",
        expected_in_answer=["pebbles"],
        description="Pebbles into cakes",
    ),
    TestCase(
        query="What did those pebbles turn into after landing on the floor?",
        expected_in_answer=["cakes", "little cakes"],
        description="Transformation of pebbles",
    ),
    TestCase(
        query="Which creature did Alice identify as Bill when she kicked down the chimney intruder?",
        expected_in_answer=["lizard", "bill"],
        description="Bill's identity",
    ),
    TestCase(
        query="What large creature did Alice compare the puppy's play to when trying not to be run over?",
        expected_in_answer=["cart-horse", "cart horse"],
        description="Puppy comparison",
    ),
    TestCase(
        query="What was the Caterpillar doing when Alice first saw it on the mushroom?",
        expected_in_answer=["smoking", "hookah"],
        description="Caterpillar first appearance",
    ),
    TestCase(
        query="Which poem did the Caterpillar ask Alice to repeat?",
        expected_in_answer=["you are old father william", "father william"],
        description="Requested recitation",
    ),
    TestCase(
        query="How tall did Alice say was 'such a wretched height' when speaking to the Caterpillar?",
        expected_in_answer=["three inches", "3 inches"],
        description="Wretched height",
    ),
    TestCase(
        query="According to the Caterpillar, what would one side of the mushroom do and what would the other side do?",
        expected_in_answer=["grow taller", "grow shorter"],
        description="Mushroom dual effect",
    ),
    TestCase(
        query="Which bird insisted that Alice must be a serpent because of her long neck?",
        expected_in_answer=["pigeon"],
        description="Serpent accusation",
    ),
    TestCase(
        query="Who brought the invitation from the Queen to the Duchess's house?",
        expected_in_answer=["fish-footman", "fish footman"],
        description="Invitation messenger",
    ),
    TestCase(
        query="What overwhelming ingredient filled the Duchess's kitchen air and soup?",
        expected_in_answer=["pepper"],
        description="Kitchen atmosphere",
    ),
    TestCase(
        query="Into what animal did the Duchess's baby finally turn?",
        expected_in_answer=["pig"],
        description="Baby transformation",
    ),
    TestCase(
        query="What direction did the Cheshire Cat give for finding the Hatter and the March Hare?",
        expected_in_answer=["that direction", "hatter", "march hare"],
        description="Route from the Cat",
    ),
    TestCase(
        query="How did the Cheshire Cat distinguish itself from a dog when arguing that it was mad?",
        expected_in_answer=["growl when i'm pleased", "wag my tail when i'm angry", "purring"],
        description="Madness logic",
    ),
    TestCase(
        query="What was unusual about the March Hare's house roof and chimneys?",
        expected_in_answer=["thatched with fur", "chimneys were shaped like ears"],
        description="March Hare house detail",
    ),
    TestCase(
        query="What drink did the March Hare offer even though none was actually on the table?",
        expected_in_answer=["wine"],
        description="Tea-party rudeness",
    ),
    TestCase(
        query="What impossible riddle did the Hatter ask Alice at the tea party?",
        expected_in_answer=["why is a raven like a writing-desk", "raven", "writing-desk"],
        description="Tea-party riddle",
    ),
    TestCase(
        query="What did the Hatter say was wrong with the watch after Alice answered the day of the month?",
        expected_in_answer=["two days wrong"],
        description="Watch complaint",
    ),
    TestCase(
        query="What did the Hatter blame for damaging the watch mechanism?",
        expected_in_answer=["butter", "crumbs"],
        description="Watch explanation",
    ),
    TestCase(
        query="At the tea party, what time did the Hatter say it always was?",
        expected_in_answer=["six o'clock", "tea-time", "tea time"],
        description="Permanent tea-time",
    ),
    TestCase(
        query="What three names did the Dormouse give for the sisters in its story?",
        expected_in_answer=["elsie", "lacie", "tillie"],
        description="Dormouse story names",
    ),
    TestCase(
        query="What did the three sisters live on at the bottom of the well?",
        expected_in_answer=["treacle"],
        description="Treacle-well story",
    ),
    TestCase(
        query="What strange M-words did the Dormouse list when describing what the sisters drew?",
        expected_in_answer=["mouse-traps", "moon", "memory", "muchness"],
        description="M-word drawing list",
    ),
    TestCase(
        query="Why were the gardeners painting the roses red?",
        expected_in_answer=["white one in by mistake", "white rose-tree", "heads cut off"],
        description="Rose painting reason",
    ),
    TestCase(
        query="What were used as the balls and mallets in the Queen's croquet game?",
        expected_in_answer=["hedgehogs", "flamingoes"],
        description="Croquet equipment",
    ),
    TestCase(
        query="What punishment did the White Rabbit say the Duchess faced, and for what reason?",
        expected_in_answer=["execution", "boxed the queen's ears"],
        description="Duchess sentence",
    ),
    TestCase(
        query="What line from Alice caused the Queen to stop listening angrily during croquet?",
        expected_in_answer=["likely to win", "hardly worth while finishing the game"],
        description="Alice self-correction before Queen",
    ),
    TestCase(
        query="What three parties argued over how to remove the Cheshire Cat's headless head problem?",
        expected_in_answer=["executioner", "king", "queen"],
        description="Cat execution dispute",
    ),
    TestCase(
        query="What did the Queen say a Mock Turtle was made from?",
        expected_in_answer=["mock turtle soup"],
        description="Mock Turtle definition",
    ),
    TestCase(
        query="Why did the Mock Turtle say their schoolmaster was called Tortoise?",
        expected_in_answer=["because he taught us"],
        description="Tortoise pun",
    ),
    TestCase(
        query="What were the first branches of arithmetic in the Mock Turtle's school?",
        expected_in_answer=["ambition", "distraction", "uglification", "derision"],
        description="Mock arithmetic",
    ),
    TestCase(
        query="What profession or creature taught Drawling, Stretching, and Fainting in Coils?",
        expected_in_answer=["conger-eel", "eel"],
        description="Drawling-master identity",
    ),
    TestCase(
        query="In the Lobster Quadrille, what happens immediately after the dancers change lobsters and retire in order?",
        expected_in_answer=["throw the lobsters", "out to sea"],
        description="Dance sequence step",
    ),
    TestCase(
        query="What song did the Mock Turtle sing just before the trial began?",
        expected_in_answer=["beautiful soup", "soup of the evening"],
        description="Mock Turtle song",
    ),
    TestCase(
        query="Who served as the judge at the trial, and what made the role look awkward?",
        expected_in_answer=["king", "crown over the wig"],
        description="Judge identity and appearance",
    ),
    TestCase(
        query="What were the jurors writing down on their slates before the trial began, according to the Gryphon?",
        expected_in_answer=["their names"],
        description="Jurors' preparation",
    ),
    TestCase(
        query="Who stole the tarts according to the accusation read aloud in court?",
        expected_in_answer=["knave of hearts"],
        description="Formal accusation",
    ),
    TestCase(
        query="What three different dates were given for when the tea had begun?",
        expected_in_answer=["fourteenth", "fifteenth", "sixteenth", "14", "15", "16"],
        description="Contradictory tea dates",
    ),
]


# ─── Document loading ──────────────────────────────────────────────────────
def load_full_document() -> str:
    return load_text(DATA_PATH)


# ─── LLM call ──────────────────────────────────────────────────────────────
def call_llm(messages):
    response = chat_completion(
        messages,
        max_tokens=256,
        temperature=0.0,
        timeout=120,
    )
    return (
        response.text,
        response.latency,
        response.prompt_tokens,
        response.completion_tokens,
        response.total_tokens,
    )


# ─── Full-context baseline ─────────────────────────────────────────────────
def run_full_context(query: str, document_text: str):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using only the provided context. "
        "If the answer is not supported by the context, say 'I don't know.' "
        "Answer briefly and directly."
    )

    user_prompt = f"""Context:
{document_text}

---
Question: {query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return call_llm(messages)


# ─── Evaluation helpers ────────────────────────────────────────────────────
def _any_keyword_match(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def check_accuracy(answer: str, expected_keywords: list[str]) -> bool:
    return _any_keyword_match(answer, expected_keywords)


def pad(s, width, align="<"):
    s = str(s)
    return s.ljust(width) if align == "<" else s.rjust(width)


# ─── Evaluation ────────────────────────────────────────────────────────────
def run_evaluation(verbose=False):
    document_text = load_full_document()

    print("\n" + "=" * 90)
    print("  No-Memory / Full-Context Evaluation  —  alice_in_wonderland.md")
    print("=" * 90)
    print(f"  Chat API  : {describe_chat_target()}")
    print("  Baseline  : Full document directly injected into prompt")
    print("=" * 90 + "\n")

    total = len(EVAL_DATASET)
    correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    total_latency = 0.0
    error_messages: list[str] = []

    rows = []

    for i, tc in enumerate(EVAL_DATASET, 1):
        try:
            answer, latency, prompt_tokens, completion_tokens, total_tokens = run_full_context(
                tc.query, document_text
            )
        except Exception as e:
            answer = f"[ERROR] {e}"
            latency = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            error_messages.append(str(e))

        is_correct = check_accuracy(answer, tc.expected_in_answer)

        if is_correct:
            correct += 1

        total_prompt_tokens += prompt_tokens or 0
        total_completion_tokens += completion_tokens or 0
        total_all_tokens += total_tokens or 0
        total_latency += latency

        rows.append({
            "id": i,
            "query": tc.query,
            "acc": "✓" if is_correct else "✗",
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        })

        if verbose:
            print(f"[{i}/{total}] {tc.query}")
            print(f"  Expected  : {', '.join(tc.expected_in_answer)}")
            print(f"  Answer    : {answer}")
            print(f"  Accuracy  : {'✓' if is_correct else '✗'}")
            print(f"  Prompt tok: {prompt_tokens}")
            print(f"  Compl tok : {completion_tokens}")
            print(f"  Total tok : {total_tokens}")
            print(f"  Latency   : {latency:.2f}s")
            print()

    accuracy = correct / total * 100
    avg_prompt_tokens = total_prompt_tokens / total if total else 0
    avg_completion_tokens = total_completion_tokens / total if total else 0
    avg_total_tokens = total_all_tokens / total if total else 0
    avg_latency = total_latency / total if total else 0

    qw = 30
    table_width = qw + 8 + 14 + 12 + 3

    print("┌" + "─" * table_width + "┐")
    header = f"│ {'#':^3}  {'Query':<{qw}}  {'Acc':^4}  {'Tokens':^10}  {'Latency':^8} │"
    print(header)
    print("├" + "─" * table_width + "┤")
    for r in rows:
        latency_text = f"{r['latency']:.2f}s"
        line = (
            f"│ {pad(str(r['id']),3)}  "
            f"{pad(r['query'],qw)}  "
            f"{pad(r['acc'],4)}  "
            f"{pad(r['total_tokens'],10, '>')}  "
            f"{pad(latency_text,8, '>')} │"
        )
        print(line)
    print("├" + "─" * table_width + "┤")
    print(f"│  Accuracy           :  {accuracy:5.1f}%  ({correct}/{total})" + " " * (table_width - 35) + "│")
    print(f"│  Avg Prompt Tokens  :  {avg_prompt_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Completion Tok :  {avg_completion_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Total Tokens   :  {avg_total_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Latency        :  {avg_latency:8.2f}s" + " " * (table_width - 35) + "│")
    print("└" + "─" * table_width + "┘")
    if error_messages:
        print(f"  Errors             : {len(error_messages)}")
        print(f"  First error        : {error_messages[0]}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_latency": avg_latency,
        "errors": error_messages,
        "rows": rows,
    }


# ─── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="No-memory / full-context baseline on alice_in_wonderland.md")
    parser.add_argument("query_text", type=str, nargs="?", help="Single query.")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details.")
    args = parser.parse_args()

    document_text = load_full_document()

    if args.eval:
        run_evaluation(verbose=args.verbose)
    elif args.query_text:
        answer, latency, prompt_tokens, completion_tokens, total_tokens = run_full_context(
            args.query_text, document_text
        )
        print("\n=== Full-Context Prompt Length Stats ===")
        print(f"Prompt Tokens     : {prompt_tokens}")
        print(f"Completion Tokens : {completion_tokens}")
        print(f"Total Tokens      : {total_tokens}")
        print(f"Latency           : {latency:.2f}s")
        print("\n=== Model Answer ===")
        print(answer)
    else:
        print("Usage:")
        print('  python query_no_memory.py "question"')
        print("  python query_no_memory.py --eval")
        print("  python query_no_memory.py --eval --verbose")


if __name__ == "__main__":
    main()
