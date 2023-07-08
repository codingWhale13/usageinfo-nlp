import csv
import pandas as pd
import os
import glob
import getch

file_path = glob.glob("sample_pairs_*")
data = pd.read_csv(file_path[0], sep=";", header=0)
# make new column with votes
data["votes"] = ""
current_index = 0


def instructions():
    clear_terminal()
    print("This script will show you pairs of usage options.")
    print("You can label them as synonyms, similar topic, different or skip them.")
    print("Press enter to continue")
    vote = input()
    if vote != "":
        exit()


def get_next_pair():
    global current_index
    if current_index < len(data):
        pair = data.iloc[current_index]
        current_index += 1
        return pair
    else:
        save_results()
        print("Labeling finished")
        exit()


def record_vote(similar):
    data["votes"][current_index - 1] = similar


def record_skip():
    data["votes"][current_index - 1] = "s"


def save_results():
    data.to_csv("sample_pairs_9_labeled.csv", sep=";", index=False)


def clear_terminal():
    if os.name == "nt":  # for Windows
        os.system("cls")
    else:  # for Linux and Mac
        os.system("clear")


def main():
    instructions()
    while True:
        clear_terminal()
        row = get_next_pair()
        while True:
            print(f"Pair {current_index} of {len(data)}")
            print(f"Usage option 1: {row[0]}")
            print(f"Usage option 2: {row[1]}")
            print("Are these usage options different? Press 1")
            print(
                "Do these usage options refer to a similar topic of use cases ? Press 2"
            )
            print(
                "Is one usage option a subset of the other/desribes the same usage option with another product? Press 3"
            )
            print(
                "Do these usage options describe the same usage option and are synonyms? Press 4"
            )
            print("Skip this pair? Press 5")
            vote = getch.getch()  # input()

            if vote == "1":
                record_vote(1)
                break
            elif vote == "2":
                record_vote(2)
                break
            elif vote == "3":
                record_vote(3)
                break
            elif vote == "4":
                record_vote(4)
                break
            elif vote == "5":
                record_vote(5)
                break
            else:
                print("Invalid input")


if __name__ == "__main__":
    main()
