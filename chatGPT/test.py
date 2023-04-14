import openai
openai.api_key = "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"
model_engine = "gpt-3.5-turbo"

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "You are determining whether a review about a professor is positive or negative."},
        {"role": "user", "content": "This was by far one of the most fun and intuitive classes I have taken here at pacific. The professors teaching style and commitment exceeded my expectations. Great class!"},
        {"role": "system", "content": "What are some keywords from the review?"},
        {"role": "user", "content": "This was by far one of the most fun and intuitive classes I have taken here at pacific. The professors teaching style and commitment exceeded my expectations. Great class!"},

    ])

message = response.choices[0]['message']

print(type(message))
print(choice)
print("{}: {}".format(message['role'], message['content']))
