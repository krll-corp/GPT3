from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('local_path', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('local_path')
tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n", tokenizer.decode(model.generate(tokenizer.encode("He is a doctor. His main goal is", return_tensors='pt'),
    max_length=128, temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3,
    num_return_sequences=1, do_sample=True)[0], skip_special_tokens=True))
