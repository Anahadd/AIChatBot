from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random
import openai

# enter the data here useful to your question. in the future, i will integrate open ai's api to generate the data for the response
data = """
"""

sent_tokens = nltk.sent_tokenize(data)

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "Not enough information, or question was not specific."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("BOT: Hi, I'm Anahad's AI Chatbot. Please enter your prompt: ")
while flag:
    user_response = input("YOU: ")
    user_response = user_response.lower()
    if user_response != 'bye':
        print("BOT: ", end="")
        print(response(user_response))
        sent_tokens.remove(user_response)
    else:
        flag = False
        print("BOT: Goodbye!")
