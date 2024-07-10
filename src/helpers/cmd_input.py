def get_yes_or_no_input(prompt, default=True) -> bool:
    user_input = None
    valid_input_options = {"y": True, "n": False, "": default}
    if default:
        prompt_postfix = "(Y/n)"
    else:
        prompt_postfix = "(y/N)"

    while user_input not in valid_input_options.keys():
        user_input = input(f"{prompt} {prompt_postfix}").strip().lower()

    return valid_input_options[user_input]
