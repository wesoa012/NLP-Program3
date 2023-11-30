def WSD_Test_Rubbish(sentences):
    output = open("./rubbish_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
        
def WSD_Test_Thorn(sentences):
    output = open("./thorn_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
        
def WSD_Test_Conviction(sentences):
    output = open("./conviction_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
