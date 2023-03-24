from nltk.corpus import wordnet

word1 = "beautiful"
word2 = "ugly"

synonyms1 = set(wordnet.synsets(word1))
synonyms2 = set(wordnet.synsets(word2))

similarity_score = 0

for syn1 in synonyms1:
    for syn2 in synonyms2:
        similarity = syn1.wup_similarity(syn2)
        if similarity is not None and similarity > similarity_score:
            similarity_score = similarity

if similarity_score > 0:
    print("The two words are similar.")
else:
    print("The two words are not similar.")
