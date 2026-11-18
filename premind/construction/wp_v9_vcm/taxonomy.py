"""Fixed fine-type taxonomy, category lexicon (I1/I2 lexicon test), visual-attribute words, names and relations.

Fine types are the enum Qwen3-VL-8B must choose from (structured output), so normalisation is by construction.
Persons are matched on (source, gender, age group) instead of a fine type (spec addendum).
"""
from __future__ import annotations

PET_TYPES = ("cat", "dog", "parrot", "other_bird", "other_animal")
OBJECT_TYPES = (
    "plush_toy", "figurine_or_action_figure", "toy_other", "toy_vehicle", "cup_or_mug", "bottle", "teapot_or_kettle",
    "bowl_or_dish", "vase", "houseplant", "flower", "candle", "lamp", "clock", "chair", "sofa", "table",
    "furniture_other", "backpack", "handbag_or_purse", "tote_bag", "wallet", "suitcase", "footwear",
    "glasses_or_sunglasses", "hat_or_helmet", "clothing", "jewelry", "watch", "headphones", "phone_or_tablet",
    "laptop", "camera", "book", "musical_instrument", "screwdriver", "hand_tool_other", "kitchen_tool",
    "sports_equipment", "bicycle", "motorbike", "car", "building", "outdoor_scene", "heater", "aerosol_can", "can",
    "cable", "power_strip", "electronics_other", "keychain", "cosmetics", "other_object")
FINE_TYPES = ("person",) + PET_TYPES + OBJECT_TYPES
GENDERS = ("male", "female", "unclear", "not_applicable")
AGE_GROUPS = ("child", "teen", "adult", "senior", "unclear", "not_applicable")
KINDS = ("person", "pet", "object")


def kind_of(fine_type: str) -> str:
    return "person" if fine_type == "person" else "pet" if fine_type in PET_TYPES else "object"


_LEX = {
    "person": "person people man woman men women boy girl kid kids child children baby toddler guy lady gentleman "
              "he she him her his hers son daughter friend brother sister mom mother dad father grandma grandpa "
              "grandmother grandfather nephew niece grandson granddaughter colleague neighbor neighbour husband wife "
              "partner teen teenager",
    "cat": "cat cats kitten kitty feline", "dog": "dog dogs puppy pup doggy hound canine pooch",
    "parrot": "parrot bird birdie budgie macaw", "other_bird": "bird birds birdie", "other_animal": "animal critter",
    "plush_toy": "plush plushie plushy stuffed teddy toy softie", "figurine_or_action_figure": "figurine statuette "
                 "collectible toy", "toy_other": "toy toys", "toy_vehicle": "toy car truck",
    "cup_or_mug": "cup cups mug mugs tumbler teacup", "bottle": "bottle bottles flask thermos",
    "teapot_or_kettle": "teapot kettle pot", "bowl_or_dish": "bowl dish plate", "vase": "vase vases",
    "houseplant": "plant plants houseplant pot succulent", "flower": "flower flowers bouquet",
    "candle": "candle candles", "lamp": "lamp", "clock": "clock", "chair": "chair seat",
    "sofa": "sofa couch", "table": "table desk", "furniture_other": "furniture cabinet shelf",
    "backpack": "backpack bag rucksack", "handbag_or_purse": "handbag purse bag", "tote_bag": "tote bag",
    "wallet": "wallet", "suitcase": "suitcase luggage", "footwear": "shoe shoes sneaker sneakers boot boots "
                                                                    "slipper slippers sandals",
    "glasses_or_sunglasses": "glasses sunglasses shades spectacles", "hat_or_helmet": "hat cap helmet",
    "clothing": "jacket coat shirt sweater dress", "jewelry": "jewelry jewellery ring earring earrings necklace",
    "watch": "wristwatch timepiece", "headphones": "headphones headset earphones", "phone_or_tablet": "phone tablet",
    "laptop": "laptop computer", "camera": "camera", "book": "book books", "musical_instrument":
    "instrument guitar violin ukulele drum", "screwdriver": "screwdriver", "hand_tool_other": "tool tools",
    "kitchen_tool": "utensil corkscrew", "sports_equipment": "ball racket bat", "bicycle": "bike bicycle",
    "motorbike": "motorbike motorcycle scooter", "car": "car vehicle", "building": "building house",
    "outdoor_scene": "garden barn", "heater": "heater", "aerosol_can": "aerosol spray canister",
    "can": "tin canister soda", "cable": "cable cord charger", "power_strip": "strip outlet", "electronics_other": "device",
    "keychain": "keychain keyring", "cosmetics": "lipstick perfume", "other_object": "",
    "toy_figure": "figurine statuette collectible doll statue toy", "cartoon_character": "character cartoon"}
FINAL_EXTRA_TYPES = ("toy_figure", "cartoon_character")   # stage-0b final taxonomy additions (finetype.py)
KIND_GENERIC = {"pet": "pet pets animal", "person": "", "object": ""}
CATEGORY_LEXICON = {t: frozenset(_LEX.get(t, "").split()) | frozenset(KIND_GENERIC[kind_of(t)].split())
                    for t in FINE_TYPES + FINAL_EXTRA_TYPES}
COLOR_WORDS = frozenset(
    "red blue green yellow orange purple violet pink brown black white gray grey beige tan cream gold golden silver "
    "navy teal turquoise maroon olive ivory khaki magenta cyan lavender".split())
PATTERN_WORDS = frozenset("striped stripes plaid checkered polka dotted spotted spots floral tartan "
                          "camo camouflage paisley".split())
ROLE_MARKERS = ("target", "confuser", "distractor", "hard negative", "decisive", "gold answer", "gt_present",
                "gt_absent", "identity_id", "concept_key")

# fixed name lists (seeded draw per (person, concept); collision-checked within a store)
PERSON_NAMES = {
    "female": "Mara Lena Iris Nora Ada Clara Elsa June Maya Tessa Vera Lucy Hana Rosa Eva Ines Dana Zoe Lila Ruth".split(),
    "male": "Leo Theo Owen Felix Hugo Milo Jonah Adam Oscar Ivan Ezra Silas Noel Abel Rafael Tomas Jude Elliot Caleb "
            "Simon".split(),
    "other": "Alex Robin Sam Jamie Taylor Rowan Quinn Avery Riley Morgan".split()}
PET_NAMES = ("Biscuit Pepper Mochi Nutmeg Ziggy Pickles Waffles Tofu Clover Juniper Pippin Marble Hazel Basil Pumpkin "
             "Peanut Sesame Toffee Maple Cocoa").split()
OBJECT_NAMES = ("Pino Whitney Courtney Amy Bram Cleo Dottie Fenna Gus Hettie Kip Lumi Mabel Nell Otto Pim Quill Rue "
                "Sable Tilly Ulla Wren Yara Zinnia Bix Coco Dex Fable Gemma Hollis").split()
RELATIONS = {("male", "child"): ("son", "nephew", "grandson"), ("female", "child"): ("daughter", "niece", "granddaughter"),
             ("male", "teen"): ("son", "nephew"), ("female", "teen"): ("daughter", "niece"),
             ("male", "adult"): ("brother", "friend", "colleague", "neighbor"),
             ("female", "adult"): ("sister", "friend", "colleague", "neighbor"),
             ("male", "senior"): ("father", "grandfather", "neighbor"),
             ("female", "senior"): ("mother", "grandmother", "neighbor")}
NEUTRAL_RELATIONS = ("friend", "neighbor", "colleague")


def relation_options(groups: set[tuple[str, str]]) -> tuple[str, ...]:
    """One shared relation for every person entry of a store (R2 B2), consistent with all their gender/age groups
    (M6); falls back to gender/age-neutral relations when the groups differ or are unclear."""
    if len(groups) == 1:
        return RELATIONS.get(next(iter(groups)), NEUTRAL_RELATIONS)
    return NEUTRAL_RELATIONS


def lexicon_for(fine_types: set[str]) -> frozenset[str]:
    out: frozenset[str] = frozenset()
    for t in fine_types:
        out |= CATEGORY_LEXICON.get(t, frozenset())
    return out


ALL_NAMES = frozenset(n.lower() for n in [*sum(PERSON_NAMES.values(), []), *PET_NAMES, *OBJECT_NAMES])
assert not ALL_NAMES & frozenset().union(*CATEGORY_LEXICON.values()), "name collides with a category word"
assert not ALL_NAMES & (COLOR_WORDS | PATTERN_WORDS), "name collides with a colour/pattern word"
