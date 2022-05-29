from typing import NamedTuple, List, Tuple

# l_i, l, r_i, r, det_idx, det_text, cent_i, cent_idx, base_concept, skeleton_id
class EntityMentionE(NamedTuple):
    l_i: int
    l: int
    r_i: int
    r: int
    det_idx: int
    det_text: str
    cent_i: int
    cent_idx: int
    base_concept: str
    skeleton_id: int

class Node(NamedTuple):
    id: int
    text: str
    words: list
    concepts: List[EntityMentionE]
    # Only meaningful in Atomic
    event_ids: list
    annotation_ids: list


class Edge(NamedTuple):
    head_id: int
    tail_id: int
    label: str
    inter_text: str
    count: int
    split: str


class Skeleton(NamedTuple):
    text: str
    subs: list # [(node_id, entity_mention_id)]