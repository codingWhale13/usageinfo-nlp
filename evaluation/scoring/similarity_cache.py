class SimilarityScorer:
    def __init__(self, comparator) -> None:
        self.comparator = comparator
        self.cache = {}

    def get(self, string1, string2, comparator=None):
        if comparator is None:
            comparator = self.comparator
        
        similarity_cache_key = self.__get_similarity_key(string1, string2)
        return self.cache[similarity_cache_key]

    def set(self, string1, string2, similarity_score, comparator=None):
        if comparator is None:
            comparator = self.comparator

        similarity_cache_key = self.__get_similarity_key(string1, string2)
        self.cache[similarity_cache_key] = similarity_score

    def __get_similarity_key(self, string1, string2):
        alphabetically_lower_string = ""
        alphabetically_higher_string = ""

        if string1 <= string2:
            alphabetically_lower_string = string1
            alphabetically_higher_string = string2
        else:
            alphabetically_lower_string = string2
            alphabetically_higher_string = string1

        return (alphabetically_lower_string, alphabetically_higher_string)

    def __calculate_score(self, string1, string2, comparator=None):
        if comparator is None:
            comparator = self.comparator
        