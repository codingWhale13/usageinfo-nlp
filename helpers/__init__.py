# sub label ids are followed with "-1", "-2", ...
MULTI_LABEL_IDS = ["bp-golden_v4"]


def get_sub_label_ids(label_id: str, all_label_ids: list[str]) -> list[str]:
    """returns all sub label ids that are part of label_id"""
    if label_id in MULTI_LABEL_IDS:
        return [sub_id for sub_id in all_label_ids if sub_id.startswith(label_id)]
    return [label_id]  # if label_id is not a multi-label; only it is relevant


__all__ = ["MULTI_LABEL_IDS", "get_sub_label_ids"]
