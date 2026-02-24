"""
Fuzzes food_ds_long.json into multiple training examples by taking every
sentence-prefix of each entry.

Data format expected in food_ds_long.json:
  {
    "sentences": ["sentence 0", "sentence 1", ...],
    "foods": [
      {
        "name": "pasta",
        "name_sentence": 1,       // 0-indexed sentence where food is first mentioned
        "initial": null           // or {"quantity": X, "unit": "Y", "sentence": Z}
                                  //   = first quantity seen (may later be corrected)
        "final": {"quantity": X, "unit": "Y", "sentence": Z}
                                  // = the correct/final quantity (null if no quantity ever stated)
      }
    ]
  }

Output format (food_ds_fuzzed.json):
  {"input": "...", "output": "food: X, quantity: Y, unit: Z|..."}

  - quantity/unit are the correct values visible at the prefix length used
  - null is used when the food is in the text but no quantity is visible yet,
    consistent with the base food_ds.json format
"""

import json
import argparse


def resolve_quantity_at_prefix(food: dict, k: int) -> tuple:
    """
    Given a food dict and prefix length k (sentences 0..k-1 are visible),
    return (quantity, unit) that should appear in the output.

    Returns (None, None) if food is visible but no quantity info is in scope yet,
    which serialises to "null" — consistent with the base food_ds.json format.
    """
    final = food.get("final")
    initial = food.get("initial")

    # Check if the final (correct) value is visible
    if final is not None and final["sentence"] < k:
        return final["quantity"], final["unit"]

    # Check if the initial (possibly wrong) value is visible but final isn't yet
    if initial is not None and initial["sentence"] < k:
        return initial["quantity"], initial["unit"]

    # Food is mentioned but no quantity info in scope
    return None, None


def format_output(visible_foods: list) -> str:
    parts = []
    for f in visible_foods:
        qty = f["quantity"]
        unit = f["unit"]
        qty_str = str(qty) if qty is not None else "null"
        unit_str = unit if unit is not None else "null"
        parts.append(f"food: {f['name']}, quantity: {qty_str}, unit: {unit_str}")
    return "|".join(parts)


def fuzz_entry(entry: dict) -> list:
    """
    Generate all prefix-length training examples for a single entry.
    Returns a list of {"input": str, "output": str} dicts.
    """
    sentences = entry["sentences"]
    foods = entry["foods"]
    n = len(sentences)
    examples = []

    for k in range(1, n + 1):
        text = " ".join(sentences[:k])

        visible_foods = []
        for food in foods:
            # Food not yet mentioned at this prefix length
            if food["name_sentence"] >= k:
                continue

            qty, unit = resolve_quantity_at_prefix(food, k)
            visible_foods.append({"name": food["name"], "quantity": qty, "unit": unit})

        if not visible_foods:
            continue

        output = format_output(visible_foods)
        examples.append({"input": text, "output": output})

    return examples


def fuzz_dataset(input_path: str, output_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    all_examples = []
    for entry in entries:
        all_examples.extend(fuzz_entry(entry))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(all_examples)} fuzzed examples from {len(entries)} entries")
    print(f"Saved to {output_path}")
    return all_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzz food_ds_long.json into training examples")
    parser.add_argument("--input", default="food_ds_long.json", help="Structured long-form dataset")
    parser.add_argument("--output", default="food_ds_fuzzed.json", help="Output training examples")
    args = parser.parse_args()

    fuzz_dataset(args.input, args.output)
