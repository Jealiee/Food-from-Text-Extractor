from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def text_to_food(text, model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    input_text = "extract_food: " + text
    print("Input text:", input_text)

    inputs = tokenizer(input_text, return_tensors='pt')
    print("Token IDs:", inputs['input_ids'])

    outputs = model.generate(**inputs, max_length=128)
    print("Generated token IDs:", outputs)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Decoded output:", generated_text)
    
    return generated_text



