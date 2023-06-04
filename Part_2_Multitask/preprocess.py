import transformers

max_length = 128

def convert_to_stsb_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    sentence1, sentence2 = example_batch['sentence1'], example_batch['sentence2']
    
    # Join the sentences together as specified in the document
    ################
    inputs = list(zip(sentence1, sentence2))
    ################ 
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, padding='max_length', truncation=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rte_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Join the sentences together as specified in the document
    ################
    sentence1, sentence2 = example_batch['sentence1'], example_batch['sentence2']
    inputs = list(zip(sentence1, sentence2))
    #features = tokenizer.batch_encode_plus(
        #inputs, max_length=max_length, pad_to_max_length=True
     #   inputs, max_length=max_length, pad_to_max_length=True, truncation=True, is_split_into_words=True, add_prefix_space=True
    #)
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    ################
    return features

def convert_to_commonsense_qa_features(example_batch, model_name="roberta-base"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    num_examples = len(example_batch["question"])
    num_choices = len(example_batch["choices"][0]["text"])
    labels2id = {char: i for i, char in enumerate("ABCDE")}
    features = {}
    
    """
    For each question:
        1) Join the same question with every answer key text. For example,
            [('Sammy wanted to go to where the people were.  Where might he go?', 'race track'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'populated areas'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'the desert'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'apartment'),
            ('Sammy wanted to go to where the people were.  Where might he go?', 'roadblock')]
        2) Then encode using batch_encode_plus
        3) Save the encoded data for corresponding keys in lists of lists for each batch
        4) Convert the answerkey to integer mapping (0 to 5) and save under the key 'labels' in features
           If answerKey does not exist, set them all to zeros
    """
    for example_i in range(num_examples):
        
        ################
        inputs = list(zip(
                [example_batch["question"][example_i]] * num_choices,
                example_batch["choices"][example_i]["text"]))
        ################
        choices_inputs = tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length, padding='max_length', truncation=True,
        )
        ################
        for key, value in choices_inputs.items():
            if key not in features:
                features[key] = []
            features[key].append(value)
    
    if example_batch["answerKey"][0]:
        features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
    else:
        features["labels"] = [0] * num_examples
        ################
    
    return features
