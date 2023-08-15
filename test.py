import openai

response = openai.Embedding.create(
    input='Create embedding for this text',
    model='text-embedding-ada-002'
)

content = response['data'][0]['embedding']

print(content)
