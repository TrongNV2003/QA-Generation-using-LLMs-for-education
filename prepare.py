def prepare_qa_input(
    t5_tokenizer,
    context,
    max_length=512,
    torch_device='cpu',
):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding.input_ids
    if torch_device == 'cuda':
        input_ids = input_ids.cuda()
    return input_ids

def prepare_distractor_input(
    t5_tokenizer,
    context,
    question,
    answer,
    separator='<sep>',
    max_length=512,
    torch_device='cpu',
):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding.input_ids
    if torch_device == 'cuda':
        input_ids = input_ids.cuda()
    return input_ids
