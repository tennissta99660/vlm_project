"""
Benchmark prompt suite: 15 prompts across 5 categories.
Each prompt has key_tokens (baseline) and compound_tokens (improved CLIPSeg queries).
"""

BENCHMARK_PROMPTS = [
    # Color-binding
    {
        "prompt": "a blue car parked next to a red fire hydrant",
        "category": "color_binding",
        "key_tokens": ["blue", "car", "red", "hydrant"],
        "compound_tokens": ["blue car", "car", "red fire hydrant", "fire hydrant"],
        "slug": "blue_car_red_hydrant",
    },
    {
        "prompt": "a green parrot sitting on a yellow branch",
        "category": "color_binding",
        "key_tokens": ["green", "parrot", "yellow", "branch"],
        "compound_tokens": ["green parrot", "parrot", "yellow branch", "branch"],
        "slug": "green_parrot_yellow_branch",
    },
    {
        "prompt": "a white cat sleeping on a black sofa",
        "category": "color_binding",
        "key_tokens": ["white", "cat", "black", "sofa"],
        "compound_tokens": ["white cat", "cat", "black sofa", "sofa"],
        "slug": "white_cat_black_sofa",
    },

    # Spatial
    {
        "prompt": "a cat sitting on top of a wooden table",
        "category": "spatial",
        "key_tokens": ["cat", "table"],
        "compound_tokens": ["cat", "wooden table"],
        "slug": "cat_on_table",
    },
    {
        "prompt": "a dog standing under a large umbrella in the rain",
        "category": "spatial",
        "key_tokens": ["dog", "umbrella", "rain"],
        "compound_tokens": ["dog", "umbrella", "rain"],
        "slug": "dog_under_umbrella",
    },
    {
        "prompt": "a bird flying above a church steeple",
        "category": "spatial",
        "key_tokens": ["bird", "church", "steeple"],
        "compound_tokens": ["bird", "church steeple", "church"],
        "slug": "bird_above_church",
    },

    # Multi-object
    {
        "prompt": "a laptop, a coffee mug, and a stack of books on a desk",
        "category": "multi_object",
        "key_tokens": ["laptop", "coffee", "mug", "books", "desk"],
        "compound_tokens": ["laptop", "coffee mug", "stack of books", "desk"],
        "slug": "laptop_mug_books_desk",
    },
    {
        "prompt": "a guitar leaning against an amplifier next to a microphone",
        "category": "multi_object",
        "key_tokens": ["guitar", "amplifier", "microphone"],
        "compound_tokens": ["guitar", "amplifier", "microphone"],
        "slug": "guitar_amplifier_mic",
    },
    {
        "prompt": "a vase of sunflowers beside a bowl of fruit on a windowsill",
        "category": "multi_object",
        "key_tokens": ["vase", "sunflowers", "bowl", "fruit", "windowsill"],
        "compound_tokens": ["vase of sunflowers", "sunflowers", "bowl of fruit", "windowsill"],
        "slug": "vase_sunflowers_fruit",
    },

    # Scene
    {
        "prompt": "a lighthouse on a rocky cliff overlooking the ocean at sunset",
        "category": "scene",
        "key_tokens": ["lighthouse", "cliff", "ocean", "sunset"],
        "compound_tokens": ["lighthouse", "rocky cliff", "ocean", "sunset sky"],
        "slug": "lighthouse_cliff_ocean",
    },
    {
        "prompt": "a cozy cabin in a snowy forest with smoke rising from the chimney",
        "category": "scene",
        "key_tokens": ["cabin", "forest", "smoke", "chimney"],
        "compound_tokens": ["cabin", "snowy forest", "smoke", "chimney"],
        "slug": "cabin_snowy_forest",
    },
    {
        "prompt": "a street market with colorful fruit stands and people walking",
        "category": "scene",
        "key_tokens": ["market", "fruit", "stands", "people"],
        "compound_tokens": ["street market", "fruit stands", "people"],
        "slug": "street_market",
    },

    # Counting
    {
        "prompt": "three red apples arranged in a row on a white plate",
        "category": "counting",
        "key_tokens": ["red", "apples", "plate"],
        "compound_tokens": ["red apples", "apples", "white plate"],
        "slug": "three_red_apples",
    },
    {
        "prompt": "two cats sitting side by side on a garden bench",
        "category": "counting",
        "key_tokens": ["cats", "bench", "garden"],
        "compound_tokens": ["cats", "garden bench"],
        "slug": "two_cats_bench",
    },
    {
        "prompt": "five candles burning on a birthday cake with frosting",
        "category": "counting",
        "key_tokens": ["candles", "cake", "frosting"],
        "compound_tokens": ["candles", "birthday cake", "frosting"],
        "slug": "five_candles_cake",
    },
]


def get_prompts_by_category(category=None):
    if category is None:
        return BENCHMARK_PROMPTS
    return [p for p in BENCHMARK_PROMPTS if p["category"] == category]


def get_categories():
    return sorted(set(p["category"] for p in BENCHMARK_PROMPTS))


def get_prompt_count():
    return len(BENCHMARK_PROMPTS)
