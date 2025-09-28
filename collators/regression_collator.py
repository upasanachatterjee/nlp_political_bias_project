import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple


class RegressionCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        # Filter out samples with invalid regression values
        valid_samples = []
        for item in batch:
            tone = item.get("V2Tone")
            parsed_value = parse_regression_values(
                str(tone) if tone is not None else None
            )
            if parsed_value is not None:
                valid_samples.append((item, parsed_value))

        # If no valid samples, return empty batch
        if not valid_samples:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "targets": torch.empty((0, 2), dtype=torch.float),
                "_skip": True,  # Indicate to skip this batch
            }

        # Extract data from valid samples only
        input_ids = [
            torch.tensor(item["input_ids"], dtype=torch.long)
            for item, _ in valid_samples
        ]
        attention_mask = [
            torch.tensor(item["attention_mask"], dtype=torch.long)
            for item, _ in valid_samples
        ]
        parsed_values = [
            torch.tensor(parsed_value, dtype=torch.float)
            for _, parsed_value in valid_samples
        ]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "targets": torch.stack(parsed_values),
        }


def parse_regression_values(row_string: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse comma-separated string and return first two values as floats."""
    if row_string is None:
        return None

    try:
        values = row_string.split(",")
        if len(values) < 2:
            return None
        return (float(values[0]), float(values[1]))
    except (ValueError, AttributeError):
        return None
