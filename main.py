from text_to_food import text_to_food

text ='Today i ate 150g of honey and a kebab. I also drank one müllermilch.'

res = text_to_food(text, r'results/checkpoint-780')

print(res)
