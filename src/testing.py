from transformers import GPT2LMHeadModel, GPT2Tokenizer
import ipdb

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
context = tokenizer('I want to fly a', return_tensors='pt')
ipdb.set_trace()
prediction = gpt2.generate(**context, max_length=10)
print("############Done############")
ipdb.set_trace()
tokenizer.decode(prediction[0])
