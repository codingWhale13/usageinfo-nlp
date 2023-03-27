from helpers.review_set import ReviewSet


json_data_1 = {
    "version": 3,
    "reviews": {
        "R1VQHRZY0310DF": {
            "marketplace": "US",
            "customer_id": "29481631",
            "product_id": "B001E8F912",
            "product_parent": "78208171",
            "product_title": "Manhattan Toy Baby Stella Black Hair Soft Nurturing First Baby Doll",
            "product_category": "Toys",
            "star_rating": 5,
            "helpful_votes": 0,
            "total_votes": 0,
            "vine": 0,
            "verified_purchase": 1,
            "review_headline": "great",
            "review_body": "My daughters love their baby doll. I use the dolls to identify body parts. They are soft and the girls like to cuddle with them when they go to sleep. They are also no too heavy for their little hands like some hard plastic baby dolls are so they tend to carry this baby doll around with them more so.",
            "review_date": "2014-02-02",
            "labels": {
                "chat_gpt-leoh_v1": {
                    "createdAt": "2023-03-14T18:43:10.173718+01:00",
                    "usageOptions": [
                        "identify body parts",
                        "cuddle with when sleeping",
                        "carry around more often",
                    ],
                    "scores": {},
                    "datasets": {},
                    "metadata": {
                        "openai": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.5,
                            "prompt_id": "leoh_v1",
                        },
                    },
                },
            },
        },
        "R221XNYDQJWR3E": {
            "marketplace": "US",
            "customer_id": "44031948",
            "product_id": "B000PEBFVO",
            "product_parent": "471382532",
            "product_title": "Avon GLAZEWEAR Liquid Lip Color",
            "product_category": "Beauty",
            "star_rating": 5,
            "helpful_votes": 0,
            "total_votes": 0,
            "vine": 0,
            "verified_purchase": 1,
            "review_headline": "pretty lip gloss",
            "review_body": "I love this lip gloss. It goes with anything I have on. It is a light pink shade that i like a lot. It isnt too glossy and is the right amount of shine.",
            "review_date": "2009-11-30",
            "labels": {
                "chat_gpt-leoh_v1": {
                    "createdAt": "2023-03-14T18:43:09.157208+01:00",
                    "usageOptions": ["beauty", "makeup"],
                    "scores": {},
                    "datasets": {},
                    "metadata": {
                        "openai": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.5,
                            "prompt_id": "leoh_v1",
                        },
                    },
                },
            },
        },
    },
}

json_data_2 = {
    "version": 3,
    "reviews": {
        "RZ4W8SDEKK2H0": {
            "marketplace": "US",
            "customer_id": "16244666",
            "product_id": "B0016CFZQ0",
            "product_parent": "935315253",
            "product_title": "Monoprice 107116 Headphone Splitter with Separate Volume Controls, White",
            "product_category": "Electronics",
            "star_rating": 5,
            "helpful_votes": 1,
            "total_votes": 1,
            "vine": 0,
            "verified_purchase": 1,
            "review_headline": "ES muy bueno",
            "review_body": "can never go wrong with monoprice. works great. love the separate volume control",
            "review_date": "2014-06-30",
            "labels": {
                "chat_gpt-leoh_v1": {
                    "createdAt": "2023-03-14T18:43:09.135598+01:00",
                    "usageOptions": [],
                    "scores": {},
                    "datasets": {},
                    "metadata": {
                        "openai": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.5,
                            "prompt_id": "leoh_v1",
                        },
                    },
                },
            },
        },
    },
}


def test_review_set():
    rs1 = ReviewSet(json_data_1)
    assert ["R1VQHRZY0310DF", "R221XNYDQJWR3E"] == sorted(rs1.reviews.keys())

    rs2 = ReviewSet(json_data_2)
    assert ["RZ4W8SDEKK2H0"] == sorted(rs2.reviews.keys())

    rs3 = rs1.merge(rs2, allow_new_reviews=True)
    assert ["R1VQHRZY0310DF", "R221XNYDQJWR3E", "RZ4W8SDEKK2H0"] == sorted(
        rs3.reviews.keys()
    )


test_review_set()
