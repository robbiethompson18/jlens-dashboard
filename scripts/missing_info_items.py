"""Paired math word problems, one complete and one with a needed quantity removed, to test whether prose lenses hallucinate
missing information. The pod writes Qwen3.6-27B's greedy reply after a shared prefill.

    uv run python scripts/missing_info_items.py   # writes data/transcripts-missing-info.json

After "Current activation oracles are hard to use" (Jakkli, Rajamanoharan, Nanda, 2026, LessWrong): on 25 such pairs their
activation oracle never named the missing quantity (0/54) and said "missing information" 59% of the time when nothing was
missing. They released no data, so these 25 pairs are new. Within a pair only the clause with the removed quantity differs;
the question sentence is the same, and both replies start with the same prefilled assistant text (PREFILL), so the readouts at
those tokens are of identical text under different internal states (the post's condition C, by construction). Thinking is
off so the prefill is the first thing the assistant says. `removed` is the ground truth: the quantity taken out (null when
complete); `answer` is the complete version's answer.
"""

import json
from pathlib import Path

PREFILL = "Let me work through this carefully, starting with what the problem gives us."
MAX_NEW_TOKENS = 450  # keeps 50 replies under ~25k tokens, so the prose readout stays near $10 of pod time

# (slug, domain, complete text, missing text, removed quantity, answer); the last sentence is the shared question
PAIRS = [
    (
        "pool volume",
        "geometry",
        "A rectangular swimming pool is 25 meters long and 10 meters wide, and it is filled to a uniform depth of 2 meters.",
        "A rectangular swimming pool is 25 meters long and 10 meters wide, and it is filled to a uniform depth.",
        "the depth (2 m)",
        "500 m³",
        "How many cubic meters of water does it hold?",
    ),
    (
        "fence perimeter",
        "geometry",
        "A rectangular garden is 12 meters long and 8 meters wide.",
        "A rectangular garden has an area of 96 square meters.",
        "the length and width (12 m by 8 m); only the area is given",
        "40 m",
        "How many meters of fencing are needed to enclose it?",
    ),
    (
        "train arrival",
        "rates",
        "A train leaves City A at 9:00 am traveling at 80 km/h toward City B, which is 360 km away.",
        "A train leaves City A at 9:00 am traveling at 80 km/h toward City B.",
        "the distance to City B (360 km)",
        "1:30 pm",
        "At what time does it arrive in City B?",
    ),
    (
        "notebook change",
        "money",
        "Maria buys 4 notebooks at $3.50 each and 3 pens at $1.20 each. She pays with a $20 bill.",
        "Maria buys 4 notebooks at $3.50 each and 3 pens. She pays with a $20 bill.",
        "the price of a pen ($1.20)",
        "$2.40",
        "How much change does she get?",
    ),
    (
        "father's age",
        "ages",
        "Tom is 3 times as old as his daughter. In 12 years, he will be twice as old as she will be then.",
        "Tom is older than his daughter. In 12 years, he will be twice as old as she will be then.",
        "the current age ratio (3 times)",
        "36",
        "How old is Tom now?",
    ),
    (
        "acid mixture",
        "mixtures",
        "A chemist mixes 6 liters of a 20% acid solution with 4 liters of a 50% acid solution.",
        "A chemist mixes 6 liters of a 20% acid solution with 4 liters of a stronger acid solution.",
        "the second solution's concentration (50%)",
        "32%",
        "What is the acid concentration of the mixture?",
    ),
    (
        "red marbles",
        "probability",
        "A bag contains 5 red marbles and 3 blue marbles. Two marbles are drawn without replacement.",
        "A bag contains 5 red marbles and some blue marbles. Two marbles are drawn without replacement.",
        "the number of blue marbles (3)",
        "5/14",
        "What is the probability that both are red?",
    ),
    (
        "two pipes",
        "work rates",
        "Pipe A can fill a tank in 6 hours, and pipe B can fill it in 3 hours.",
        "Pipe A can fill a tank in 6 hours, and pipe B can fill it faster.",
        "pipe B's fill time (3 h)",
        "2 hours",
        "If both pipes are opened together, how long does it take to fill the tank?",
    ),
    (
        "simple interest",
        "money",
        "Alex deposits $2,000 in an account that pays 5% simple interest per year.",
        "Alex deposits $2,000 in an account that pays simple interest every year.",
        "the interest rate (5%)",
        "$300",
        "How much interest has he earned after 3 years?",
    ),
    (
        "garden path",
        "geometry",
        "A circular garden has a radius of 7 meters. A path 1 meter wide runs around the outside of the garden.",
        "A circular garden has a radius of 7 meters. A path runs around the outside of the garden.",
        "the path's width (1 m)",
        "15π ≈ 47.1 m²",
        "What is the area of the path?",
    ),
    (
        "average speed",
        "rates",
        "A cyclist rides 30 km at 15 km/h and then another 30 km at 10 km/h.",
        "A cyclist rides 30 km at 15 km/h and then another 30 km at a slower speed.",
        "the second leg's speed (10 km/h)",
        "12 km/h",
        "What is her average speed for the whole trip?",
    ),
    (
        "jacket price",
        "percentages",
        "A jacket originally costs $80. It is discounted by 25%, and then a 10% sales tax is added to the discounted price.",
        "A jacket originally costs $80. It is discounted, and then a 10% sales tax is added to the discounted price.",
        "the discount (25%)",
        "$66",
        "What is the final price?",
    ),
    (
        "right triangle",
        "geometry",
        "A right triangle has legs of length 9 cm and 12 cm.",
        "A right triangle has one leg of length 9 cm.",
        "the other leg (12 cm)",
        "36 cm",
        "What is its perimeter?",
    ),
    (
        "fuel cost",
        "rates",
        "A car uses 6 liters of fuel per 100 km, and fuel costs $1.80 per liter.",
        "A car uses 6 liters of fuel per 100 km, and fuel prices vary by station.",
        "the fuel price ($1.80/L)",
        "$48.60",
        "How much does the fuel cost for a 450 km trip?",
    ),
    (
        "recipe scaling",
        "ratios",
        "A recipe that serves 4 people uses 300 grams of flour.",
        "A recipe uses 300 grams of flour.",
        "how many people the recipe serves (4)",
        "750 g",
        "How much flour is needed to make it for 10 people?",
    ),
    (
        "sibling ages",
        "ages",
        "The sum of Anna's and Ben's ages is 30, and Anna is 4 years older than Ben.",
        "The sum of Anna's and Ben's ages is 30, and Anna is older than Ben.",
        "the age difference (4 years)",
        "13",
        "How old is Ben?",
    ),
    (
        "slow clock",
        "time",
        "A clock loses 2 minutes every hour. It is set to the correct time at noon.",
        "A clock loses a few minutes every hour. It is set to the correct time at noon.",
        "the rate it loses time (2 min/h)",
        "5:48 pm",
        "What time will it show when the correct time is 6:00 pm the same day?",
    ),
    (
        "cylinder tank",
        "geometry",
        "A cylindrical tank has a radius of 2 meters and a height of 5 meters.",
        "A cylindrical tank has a height of 5 meters.",
        "the radius (2 m)",
        "20π m³ ≈ 62,832 L",
        "How many liters of water does it hold when full?",
    ),
    (
        "blue spinner",
        "probability",
        "A spinner is divided into 8 equal sections, 3 of which are blue.",
        "A spinner is divided into 8 equal sections, some of which are blue.",
        "the number of blue sections (3)",
        "9/64",
        "If it is spun twice, what is the probability that it lands on blue both times?",
    ),
    (
        "town growth",
        "percentages",
        "A town has 12,000 residents, and its population grows by 5% each year.",
        "A town has 12,000 residents, and its population grows by a fixed percentage each year.",
        "the growth rate (5%)",
        "13,230",
        "What will its population be after 2 years?",
    ),
    (
        "cars meeting",
        "rates",
        "Two cars start 300 km apart and drive toward each other, one at 70 km/h and the other at 80 km/h.",
        "Two cars start some distance apart and drive toward each other, one at 70 km/h and the other at 80 km/h.",
        "the starting distance (300 km)",
        "2 hours",
        "How long until they meet?",
    ),
    (
        "lamp profit",
        "money",
        "A store buys 50 lamps for $24 each and sells all of them for $39 each.",
        "A store buys 50 lamps for $24 each and sells all of them at a markup.",
        "the selling price ($39)",
        "$750",
        "What is the store's total profit?",
    ),
    (
        "class ratio",
        "ratios",
        "A class has 28 students, and the ratio of girls to boys is 4:3.",
        "A class has 28 students, and there are more girls than boys.",
        "the girls-to-boys ratio (4:3)",
        "16",
        "How many girls are in the class?",
    ),
    (
        "wall paint",
        "geometry",
        "A room is 5 meters long, 4 meters wide and 3 meters high. One liter of paint covers 10 square meters.",
        "A room is 5 meters long and 4 meters wide. One liter of paint covers 10 square meters.",
        "the room's height (3 m)",
        "5.4 L",
        "How many liters are needed to paint the four walls, ignoring doors and windows?",
    ),
    (
        "phone bill",
        "money",
        "A phone plan charges a $15 monthly fee plus $0.10 per minute of calls. Last month Sam made 240 minutes of calls.",
        "A phone plan charges a $15 monthly fee plus $0.10 per minute of calls. Last month Sam made several calls.",
        "the minutes of calls (240)",
        "$39",
        "What was his bill last month?",
    ),
]

items = []
for n, (slug, domain, complete, missing, removed, answer, question) in enumerate(PAIRS, 1):
    for version, setup in (("complete", complete), ("missing", missing)):
        prompt = f"{setup} {question}"
        items.append(
            {
                "id": f"mi-{n:02d}-{version}",
                "name": f"{n:02d} {slug}: " + ("complete" if version == "complete" else f"missing {removed}"),
                "source": version,
                "think": False,
                "prompt": prompt,
                "prefill": PREFILL,
                "category": domain,
                "removed": None if version == "complete" else removed,
                "answer": answer if version == "complete" else None,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dataset_split": f"mi-{n:02d}",
            }
        )
Path("data/transcripts-missing-info.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
print(f"wrote {len(items)} items ({len(PAIRS)} pairs)")
