import tkinter as tk

pairs = [("cat", "dog"), ("apple", "banana"), ("car", "bicycle")]
current_pair_index = 0


def vote_similar():
    print("Similar")
    next_pair()


def vote_not_similar():
    print("Not similar")
    next_pair()


def next_pair():
    global current_pair_index
    current_pair_index += 1

    if current_pair_index < len(pairs):
        pair_label.config(text=f"Pair: {pairs[current_pair_index]}")
    else:
        pair_label.config(text="No more pairs!")


root = tk.Tk()
root.title("Word Similarity Labeling App")

pair_label = tk.Label(root, text=f"Pair: {pairs[current_pair_index]}")
pair_label.pack(pady=10)

similar_button = tk.Button(root, text="Similar", command=vote_similar)
similar_button.pack(pady=5)

not_similar_button = tk.Button(root, text="Not Similar", command=vote_not_similar)
not_similar_button.pack(pady=5)

root.mainloop()
