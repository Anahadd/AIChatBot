from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pygame
from pygame.locals import *
import time
import random
import openai

# enter the data here useful to your question. in the future, i will integrate open ai's api to generate the data for the response
data = """
Cookies are small text files that are stored on a user's device when they visit a website. They often contain information about user preferences, login details, and tracking information. Cookies can be categorized into session cookies, persistent cookies, and third-party cookies. Session cookies are temporary and deleted after the browser is closed, while persistent cookies are stored between sessions, and third-party cookies are set by domains other than the one being visited. Cookies are used for authentication, personalization, tracking, and analytics. Different regions have regulations surrounding the use of cookies, and consent is often required. Cookies in baking refer to sweet baked treats made with ingredients like flour, sugar, butter, eggs, and flavorings such as chocolate chips or nuts. They come in various types and are culturally significant in many traditions.
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

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Anahad\'s AI Chatbot')
font = pygame.font.Font(None, 32)
clock = pygame.time.Clock()
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color_white = (255, 255, 255)
input_box = pygame.Rect(100, 500, 600, 32)
color = color_inactive
text = ''
active = False
conversation = ["BOT: Hi, I'm Anahad's AI Chatbot. Please enter your prompt: "]

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
        if event.type == MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                active = not active
            else:
                active = False
            color = color_active if active else color_inactive
        if event.type == KEYDOWN:
            if active:
                if event.key == K_RETURN:
                    user_response = text.lower()
                    if user_response != 'bye':
                        conversation.append("YOU: " + user_response)
                        bot_response = response(user_response)
                        conversation.append("BOT: " + bot_response)
                        sent_tokens.remove(user_response)
                        text = ''
                    else:
                        conversation.append("YOU: " + user_response)
                        conversation.append("BOT: Goodbye!")
                        text = ''
                        time.sleep(2)
                        pygame.quit()
                        exit()
                elif event.key == K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode

    txt_surface = font.render(text, True, color)
    width = max(600, txt_surface.get_width() + 10)
    input_box.w = width
    screen.fill(color_white)
    screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
    pygame.draw.rect(screen, color, input_box, 2)

    y_pos = 10
    for line in conversation[-15:]:
        line_surface = font.render(line, True, (0, 0, 0))
        screen.blit(line_surface, (10, y_pos))
        y_pos += 30

    pygame.display.flip()
    clock.tick(30)
