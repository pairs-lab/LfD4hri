"""Public task-language grammar; no simulator poses/images/evaluation seeds."""
import hashlib
import json

from config import INSTRUCTION_FORM, PICK_TARGETS, PLACE_TARGETS


def declared_prompt_vocabulary():
    prompts = {template.format(pick) for template in INSTRUCTION_FORM[:2] for pick in PICK_TARGETS}
    prompts.update(template.format(pick, place) for template in INSTRUCTION_FORM[2:]
                   for pick in PICK_TARGETS for place in PLACE_TARGETS if pick != place)
    return sorted(prompts)


def vocabulary_sha256():
    return hashlib.sha256(json.dumps(declared_prompt_vocabulary(), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()
