import gensim.downloader as api
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()  

print("Enter two space-separated words")
words = input()
  
tokens = nlp(words)
  
for token in tokens:
    # Printing the following attributes of each token.
    # text: the word string, has_vector: if it contains
    # a vector representation in the model, 
    # vector_norm: the algebraic norm of the vector,
    # is_oov: if the word is out of vocabulary.
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
  
token1, token2 = tokens[0], tokens[1]
  
print("Similarity:", token1.similarity(token2))
#model = api.load('word2vec-google-news-300')
#similarity_score = []
#similarity_score.append(model.similarity('happy', 'cheery'))
#similarity_score.append(model.similarity('happy', 'jolly'))
#similarity_score.append(model.similarity('happy', 'ecstatic'))
#similarity_score.append(model.similarity('happy', 'unhappy'))
#similarity_score.append(model.similarity('happy', 'depressed'))
#similarity_score.append(model.similarity('happy', 'sad'))

#print(similarity_score)
