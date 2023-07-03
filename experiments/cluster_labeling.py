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
    print("Press 'y' if you think they should appear in the same cluster.")
    print("Press 'n' if the usage options are not similar at all.")
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
    data.to_csv("sample_labeled.csv", sep=";", index=False)


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
            print(f"Usage option 1: {row[0]}")
            print(f"Usage option 2: {row[1]}")
            print(
                "Do these usage options describe the same usage option and should therefore appear in the same cluster? (y/n) (s to skip)"
            )
            vote = getch.getch()  # input()

            if vote == "y":
                record_vote(True)
                break
            elif vote == "n":
                record_vote(False)
                break
            elif vote == "s":
                record_skip()
                break
            else:
                print("Invalid input")


if __name__ == "__main__":
    main()
