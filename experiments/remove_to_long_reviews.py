from helpers.review_set import ReviewSet

REVIEWS = [
    "R16HNWNTJHSUN",
    "R2319KRS3Z356Q",
    "R3DOEAKJZNNRD7",
    "R3P1Q67E05S6JW",
    "R1EMURKJDD93O9",
    "RVHMIBFXTHC38",
    "R2VJ5MUZQOE54X",
    "R1D31Z4WXCPWMU",
    "R1HVPPTLRPPD1Q",
    "RV89FCENVXWVQ",
    "RVVE9Z5WY6RCV",
    "R1ZH7MBCLUTN24",
    "R2SC8QAV3HFDBD",
    "R1YLZY1U721O1K",
    "R3BGZLRXUO7L3F",
    "R2GBR8BJQO9MHF",
    "R3EO21RY51RJV7",
    "RTTN0XP1HZ2B6",
    "R1CTTM1P15CR3K",
    "R3TE1QM8REICGP",
    "R3EDBVHW9U7KH",
    "R2BC46OZ3Q9KCV",
    "R15YUN69H97M14",
    "RIWHMMXAQQSCH",
    "RTBS6SD8UNNMJ",
    "R3H0B61CWNXJEK",
    "R1KSU05K9MM8BA",
    "R2COGXHWJ2QOAF",
    "R1RQ4Q5MIPF2JH",
    "RTYADAGNF9L8Q",
    "R2ITE5BFPFSUH4",
    "R2LW0JEKR26W5P",
    "R2950JSEB8BSVO",
    "R1JNFKKOWK7MGF",
    "R3AG2U3UMVM9O9",
    "R2QJ99UMTIRVG6",
    "RE75TJBD8AB28",
    "R1P0CVHXZ84A7F",
    "R2ZUZJ2WKROWTF",
    "RXVFZ6P1TG7OZ",
    "R1PRDTN17H4MR4",
    "R165GSNXK6LUNM",
    "R3MSWAX61BGSJW",
    "R2CODUS8QSMMD9",
    "R1OC4EGSR0H806",
    "R2R0TW6I3R7L45",
    "R2811PPDCTB8XM",
    "R1269RP6A8G9M3",
    "RW6IVQZNZJ087",
    "R30V51U4VYO553",
    "R2EP76HH7EJVC1",
    "R3UCZW8EWG82FA",
    "R19ZK1N2438XHX",
    "R30VHU5MOPP5SJ",
    "R3PCD0QC79DS95",
]
FILE = "../../ba/few-shot/jsons/flan-t5-base_without_too_long.json"

rs = ReviewSet.from_files(FILE)
new_reviews = {}
for review_id, review in rs.reviews.items():
    if review_id not in REVIEWS:
        new_reviews[review_id] = review

rs.reviews = new_reviews
rs.save()
