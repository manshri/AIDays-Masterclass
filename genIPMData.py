import random

# Expanded lists of iambic-friendly words
nouns = [
    "heart", "dream", "light", "night", "soul", "time", "star", "world", "mind", "love",
    "sky", "wind", "fire", "hope", "grace", "truth", "shade", "flame", "voice", "sea"
]
verbs = [
    "shines", "flows", "burns", "cries", "runs", "feels", "waits", "holds", "knows", "calls",
    "fades", "breaks", "rises", "falls", "sings", "moves", "dreams", "sleeps", "speaks", "walks"
]
adjectives = [
    "soft", "bright", "dark", "deep", "still", "lost", "bold", "pure", "cold", "warm",
    "clear", "wild", "faint", "slow", "sweet", "calm", "lone", "sharp", "small", "brave"
]
prepositions = [
    "above", "within", "beyond", "beneath", "before", "around", "upon", "against", "between", "through",
    "inside", "outside", "near", "under", "over"
]
articles = ["the", "a", "my", "your", "his", "her", "its", "our", "their", "this"]

# Expanded templates for more variety
templates = [
    "The {adj} {noun} {verb} {prep} the {adj} {noun}.",
    "I {verb} the {adj} {noun} {prep} {art} {noun}.",
    "To {verb} {art} {adj} {noun} is {art} {noun}.",
    "{art} {noun} {verb} {prep} {art} {adj} {noun}.",
    "{art} {adj} {noun} {verb} in {art} {noun}.",
    "{art} {noun} {verb} and {art} {noun} {verb}.",
    "She {verb} {art} {noun} {prep} the {adj} {noun}.",
    "To {verb} and {verb} is all the {noun} does.",
    "{art} {adj} {noun} will {verb} with {art} {noun}.",
    "He {verb} {prep} the {adj} {noun} at {art} {noun}."
]


def generate_line():
    template = random.choice(templates)
    return template.format(
        noun=random.choice(nouns),
        verb=random.choice(verbs),
        adj=random.choice(adjectives),
        prep=random.choice(prepositions),
        art=random.choice(articles)
    )

# Generate 1000 lines
lines = [generate_line() for _ in range(5000)]
poem = "\n".join(lines)
poem[:5000]  # Preview only first 1000 characters of the poem

# Save the poem to a file
with open("iambic_eval_data.txt", "w") as f:
    f.write(poem)

print("Poem saved to iambic_poem.txt")
