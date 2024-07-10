import os
import pickle

CACHE_PATH = "/hpi/fs00/share/fg-demelo/bsc2022-usageinfo/evaluation/caches/similarity_cache.pickle"


class EvaluationCache(dict):
    """This is a singleton, keeping track of token embeddings and OpenAI phrase similarities.

    Why cache embeddings? To avoid overhead when calling model during scoring or clustering.
    Why cache OpenAI phrase similarities? To reduce API requests and save money.

    Key format: (comparator, [XYZ])
    [XYZ] for comparator == "openai" are the following strings:
        - phrase_1 (lexicographically smaller)
        - phrase_2 (lexicographically larger)
        - OpenAI model
        - prompt ID
        - temperature
    [XYZ] for comparator != "openai" are the following strings:
        - phrase (i.e. usage option)

    Value:
        - for comparator == "openai": similarity class (as string)
        - for comparator != "openai": embedding output from the respective model (spacy or all-mpnet-base-v2)
    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(EvaluationCache, cls).__new__(cls)
            cls.instance.load_from_disk()
        return cls.instance

    @classmethod
    def get(cls):
        return cls()

    def load_from_disk(self):
        try:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "rb") as file:
                    self.update(pickle.load(file))
            else:
                with open(CACHE_PATH, "wb") as file:
                    pickle.dump({}, file)
        except Exception:
            print(
                f"WARNING: failed to load similarity cache from disk. Make sure you're on the cluster to use it!"
            )

    def save_to_disk(self):
        try:
            self.load_from_disk()  # make sure we didn't miss anything

            print("saving similarity cache to disk...")
            with open(CACHE_PATH, "wb") as file:
                pickle.dump(self.copy(), file)
        except Exception:
            print(f"WARNING: could not save evaluation cache in {CACHE_PATH}")
