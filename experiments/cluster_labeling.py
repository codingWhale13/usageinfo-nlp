import csv
import pandas as pd
import os

pairs = [("a", "b"), ("c", "d"), ("e", "f")]
current_pair_index = 0
results = []


def instructions():
    clear_terminal()
    print("This script will show you pairs of usage options.")
    print("Press 'y' if you think they should appear in the same cluster.")
    print("Press 'n' if the usage options are not similar at all.")
    print("Press enter to continue")
    vote = input()
    if vote != "":
        exit()


def get_next_pair():
    global pairs
    global current_pair_index
    if current_pair_index < len(pairs):
        pair = pairs[current_pair_index]
        current_pair_index += 1
        return pair
    else:
        save_results()
        print("Labeling finished")
        exit()


def record_vote(similar):
    global results
    pair = pairs[current_pair_index - 1]
    results.append([pair[0], pair[1], similar])


def save_results():
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)


def clear_terminal():
    if os.name == "nt":  # for Windows
        os.system("cls")
    else:  # for Linux and Mac
        os.system("clear")


def main():
    instructions()
    while True:
        clear_terminal()
        pair = get_next_pair()
        while True:
            print(f"Usage option 1: {pair[0]}")
            print(f"Usage option 2: {pair[1]}")
            print(
                "Do these usage options describe the same usage option and should therefore appear in the same cluster? (y/n)"
            )
            vote = input()

            if vote == "y":
                record_vote(True)
                break
            elif vote == "n":
                record_vote(False)
                break
            else:
                print("Invalid input")


if __name__ == "__main__":
    main()
