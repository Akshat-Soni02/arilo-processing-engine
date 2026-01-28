from config.config import (
    Llm_Call,
    User_Input_Type,
    Plan_Type,
    LLM_CONFIG,
)
from common.utils import read_file
from typing import Optional, Sequence


def get_llm_input(
    llm_call: Llm_Call,
    input: Optional[bytes] = None,
    input_type: Optional[User_Input_Type] = None,
    replace: Optional[Sequence[dict]] = None,
    plan_type: Optional[Plan_Type] = None,
):
    """
    Get the structured LLM input dictionary based on call type and plan.
    """
    # Default to FREE plan if not specified or invalid
    if not plan_type or plan_type not in LLM_CONFIG:
        plan_type = Plan_Type.FREE

    config = LLM_CONFIG[plan_type].get(llm_call)
    if not config:
        return None

    return prepare_llm_input(config, input, input_type, replace)


def prepare_llm_input(
    config: dict,
    input: Optional[bytes] = None,
    input_type: Optional[User_Input_Type] = None,
    replace: Optional[Sequence[dict]] = None,
) -> dict:
    """
    Take configuration dictionary and build the actual LLM input.
    """
    prompt_file_path = config.get("PROMPT_FILE_PATH")
    system_instruction_file_path = config.get("SYSTEM_INSTRUCTION_FILE_PATH")
    response_schema_file_path = config.get("RESPONSE_SCHEMA_FILE_PATH")

    prompt = None
    system_instruction = None
    response_schema = None

    if prompt_file_path:
        prompt = read_file(prompt_file_path)
    if system_instruction_file_path:
        system_instruction = read_file(system_instruction_file_path)
    if response_schema_file_path:
        response_schema = read_file(response_schema_file_path, is_json=True)

    if replace:
        for item in replace:
            if item["type"] == "prompt":
                if prompt is None:
                    continue
                prompt = prompt.replace(item["replace_key"], item["replace_value"])
            elif item["type"] == "sys":
                if system_instruction is None:
                    continue
                system_instruction = system_instruction.replace(
                    item["replace_key"], item["replace_value"]
                )

    llm_input = {
        "model": config.get("MODEL"),
        "token_limit": config.get("TOKEN_LIMIT"),
        "prompt": prompt,
        "system_instruction": system_instruction,
        "response_schema": response_schema,
    }

    if input and input_type:
        llm_input["input_type"] = input_type
        llm_input["user_data"] = input

    return llm_input
