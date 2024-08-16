from transformers import pipeline

#Download BERT Model and tokenizer
classfier = pipeline('sentiment-analysis', "bert-base-uncased")

#Converting LABEL_1 and LABEL_0 to positive and negative
label_map = {
    'LABEL_0': 'NEGATIVE',
    'LABEL_1': 'POSITIVE'

}
#Test Sentence
text = "Is this a new hat?"

#Analysis
result = classfier(text)
#Converting the result
result[0]['label'] = label_map[result[0]['label']]
#Print the results
print(result)