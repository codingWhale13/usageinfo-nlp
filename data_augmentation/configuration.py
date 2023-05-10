import data_augmentation.core as da_core
from data_augmentation.back_translation import BackTranslationTextAugmentation
from data_augmentation.core import TestTextAugmentation

BC_BOLD = "\033[1m"
BC_ENDC = "\033[0m"


def _get_input(
    options: list[str],
    prompt: str = f"{BC_BOLD}Select one of the following options:{BC_ENDC}",
    default: int = None,
) -> int:
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"({i}) {option} {'-> [default]' if i == default else ''}")
    print()

    selection = int(input("Enter your selection: ") or default) - 1
    if selection not in range(len(options)):
        raise ValueError("Choose one of the options above!")
    print()
    return options[selection]


def _handle_wrong_input(error: ValueError):
    print("Invalid selection!")
    print(error)


def get_back_translation_augmentation():
    translation_rounds = None
    while not translation_rounds:
        try:
            translation_rounds = int(
                input(
                    f"{BC_BOLD}Enter{BC_ENDC} the {BC_BOLD}number of translation rounds{BC_ENDC} -> [default: 1 round]: "
                )
                or 1
            )
            if translation_rounds < 1:
                translation_rounds = None
                raise ValueError("Translation rounds must be at least 1!")
        except ValueError as e:
            _handle_wrong_input(e)

    return BackTranslationTextAugmentation(translation_rounds=translation_rounds)


def get_test_text_augmenation():
    return TestTextAugmentation()


text_augmentations = {
    "BackTranslationTextAugmentation": get_back_translation_augmentation,
    "TestTextAugmentation": get_test_text_augmenation,
}


def get_text_augmentation():
    text_augmentation = None
    while not text_augmentation:
        try:
            selection = _get_input(
                list(text_augmentations.keys()),
                f"{BC_BOLD}Select{BC_ENDC} which {BC_BOLD}text augmentation{BC_ENDC} to apply:",
                default=1,
            )
            text_augmentation = text_augmentations[selection]()

        except ValueError as e:
            _handle_wrong_input(e)
    return text_augmentation, {selection: text_augmentation.metadata()}


def get_partial_review_augmentation() -> (
    tuple[da_core.PartialReviewAugementation, dict]
):
    augmented_parts = None
    while not augmented_parts:
        try:
            options = [
                ["review_body"],
                ["usageOptions"],
                ["product_title"],
                ["review_body", "usageOptions"],
                ["review_body", "product_title"],
                ["usageOptions", "product_title"],
                ["review_body", "usageOptions", "product_title"],
            ]
            augmented_parts = _get_input(
                options,
                f"{BC_BOLD}Select{BC_ENDC} which {BC_BOLD}part(s) of the review{BC_ENDC} should be augmented:",
                default=4,
            )

        except ValueError as e:
            _handle_wrong_input(e)

    text_augmentation, text_aug_config = get_text_augmentation()
    return (
        da_core.PartialReviewAugementation(text_augmentation, augmented_parts),
        {"text_augmentation": text_aug_config, "augmented_parts": augmented_parts},
    )


review_augmentations = {
    "PartialReviewAugmentation": get_partial_review_augmentation,
}


def get_augmentations() -> tuple[da_core.MultiAugmentation, list[tuple[str, dict]]]:
    augmentations = []
    augmentation_config = {}
    i = 1
    while True:
        try:
            options = list(review_augmentations.keys()) + ["Exit"]
            selection = _get_input(
                options,
                f"{BC_BOLD}Select {i}. augmentation{BC_ENDC} or {BC_BOLD}exit{BC_ENDC}:",
                default=len(options),
            )

            if selection == "Exit":
                break
            augmentation, config = review_augmentations[selection]()
            augmentations.append(augmentation)
            augmentation_config[f"{i}_{selection}"] = config

            i += 1
        except ValueError as e:
            _handle_wrong_input(e)

    return da_core.MultiAugmentation(*augmentations), augmentation_config
