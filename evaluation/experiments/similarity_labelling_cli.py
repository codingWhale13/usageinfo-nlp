# this script should start an interactive session to similarity a given set of pairs based on their similarity

import argparse
import pickle
import os
import progressbar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score usage option pairs based on their similarity."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the usage option pairs to similarity.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output file to store the similarityd pairs.",
    )
    parser.add_argument(
        "-v",
        "--verbse",
        action="store_true",
        help="Print scores for each pair after annotating.",
    )
    return parser.parse_args()


def main():
    # load pickled pairs
    args = parse_args()

    with open(args.input, "rb") as f:
        pairs = pickle.load(f)

    # print task
    os.system("clear")
    print(
        """
    Your task is to determine the similarity of the usage option pairs.
    The similarity should be a number between between 0 and 10, where 0 means not similar at all and 10 means they are identical.
          
    As an orientation, you can be inspired by the following examples:
          
        1. (yard work, working in the garden) 
        -> 10, because they are identical
        
        2. (blending fruits, blending vegetables) 
        -> 9, because they are very similar
        
        3. (great for funerals, good for weddings) 
        -> 4, because they are not very similar but both are related to events
        
        4. (working out, taking study notes) 
        -> 0, because they are not similar at all
          """
    )

    input("Press enter to continue.")

    # ask the user to write a similarity_id which will be used to identify the similarity
    similarity_set = False
    similarity_id = None
    operation = None

    while not similarity_set:
        os.system("clear")
        print("Please enter a similarity ID to identify the similarity.")
        similarity_id = input("Similarity ID: ")

        if similarity_id == "":
            print("Similarity ID cannot be empty.")
            input("Press enter to retry.")
        elif similarity_id in [
            "all-mpnet-base-v2",
            "bleu",
            "sacrebleu",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
            "openai",
        ]:
            print(
                f"Similarity ID cannot be '{similarity_id}', since it is a reserved similarity ID."
            )
            input("Press enter to retry.")
        else:
            # print warning, if similarity_id already exists in any of the keys in pairs; print error if exists
            similarity_id_exists = False
            for pair in pairs.values():
                if similarity_id in pair["similarities"].keys():
                    print(f"Similarity ID already exists in pair: {similarity_id}.")
                    similarity_id_exists = True

                    # make user chose mode of operation: continue labelling, overwrite or cancel
                    print(
                        """
        Please choose a mode of operation (enter the number)):
        1. Continue labelling
        2. Overwrite existing similarity
        3. Cancel
        """
                    )
                    while True:
                        operation = input("Operation: ")
                        if operation != "1" and operation != "2" and operation != "3":
                            print("Invalid operation.")
                            input("Press enter to restart.")
                            print(
                                "\033[A                                              \033[A"
                            )
                            print(
                                "\033[A                                              \033[A"
                            )
                            print(
                                "\033[A                                              \033[A"
                            )
                        elif operation == "3":
                            exit()
                        else:
                            break
                    similarity_set = True
                    break

            if not similarity_id_exists:
                # ask user to confirm similarity_id; y be default and can be selected by pressing enter
                confirm = (
                    input(f"Confirm similarity ID: {similarity_id} [[y]/n]: ") or "y"
                )
                if confirm == "y":
                    similarity_set = True
                    print(f"Similarity ID set to: {similarity_id}")
                else:
                    print("Similarity ID not set.")
                    input("Press enter to restart.")

    # similarity pairs

    os.system("clear")

    bar = progressbar.ProgressBar(
        maxval=len(pairs),
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for key, value in pairs.items():
        similarity_set = False
        if operation != "1" or not similarity_id in value["similarities"]:
            bar.update(list(pairs.keys()).index(key))
            print()
            print("Usage options: ")
            print("1. ", key[0])
            print("2. ", key[1])
            if similarity_id in value["similarities"]:
                print(
                    f"Current similarity: {int(value['similarities'][similarity_id] * 10)}"
                )

            while not similarity_set:
                similarity = input("Similarity: ")

                # check if similarity is a number between 0 and 10
                try:
                    if similarity != "" or not similarity_id in value["similarities"]:
                        similarity = int(similarity)
                        if similarity < 0 or similarity > 10:
                            raise ValueError
                        value["similarities"][similarity_id] = similarity / 10.0

                        if args.verbse:
                            print("----------------------------------------")
                            print("Similarities:")
                            for metric, value in value["similarities"].items():
                                print(f"{metric}: {value}")

                            print("----------------------------------------")
                            input("Press enter to continue.")
                    # save pairs

                    with open(f"{args.output}", "wb") as f:
                        pickle.dump(pairs, f)

                    similarity_set = True
                    os.system("clear")
                except ValueError:
                    print("Similarity must be a number between 0 and 10.")
                    input("Press enter to label again.")
                    print("\033[A                                              \033[A")
                    print("\033[A                                              \033[A")
                    print("\033[A                                              \033[A")

    bar.finish()


if __name__ == "__main__":
    main()
