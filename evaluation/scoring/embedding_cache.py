class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class EmbeddingCache(Singleton):
    """Cache for embeddings.

    Parameters
    ----------
    embeddings : Embeddings
    Embeddings to cache.
    """

    cache = {}

    def load(self, string, str_sim):
        return self.cache[(string, str_sim)]

    def add(self, string, str_sim, value):
        self.cache[(string, str_sim)] = value

    def exists(self, string, str_sim):
        return (string, str_sim) in self.cache

    def __len__(self):
        return len(self.cache)

    @classmethod
    def get(cls):
        return EmbeddingCache()
