import csv
import pandas as pd

pairs = [("a", "b"), ("c", "d"), ("e", "f")]
current_pair_index = 0
results = []


def get_next_pair():
    global pairs
    global current_pair_index
    if current_pair_index < len(pairs):
        print(f"Pair: {pairs[current_pair_index]}")
        current_pair_index += 1
    else:
        save_results()
        print("Labeling finished")
        exit()


def vote_similar():
    record_vote(True)
    get_next_pair()


def vote_not_similar():
    record_vote(False)
    get_next_pair()


def record_vote(similar):
    global results
    pair = pairs[current_pair_index - 1]
    results.append([pair[0], pair[1], similar])


def save_results():
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)


def main():
    get_next_pair()
    while True:
        print(
            "Would you consider these usage options to appear in the same cluster? (y/n)"
        )
        vote = input()
        if vote == "y":
            vote_similar()
        elif vote == "n":
            vote_not_similar()
        else:
            print("Invalid input")


if __name__ == "__main__":
    main()
