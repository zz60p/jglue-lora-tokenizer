from datasets import load_dataset

def load_jnli(split='validation'):
    ds = load_dataset('shunk031/JGLUE', 'JNLI', split=split)
    # columns: 'sentence1', 'sentence2', 'label'
    return ds

def load_jsquad(split='validation'):
    ds = load_dataset('shunk031/JGLUE', 'JSQuAD', split=split)
    # fields: 'id','title','context','question','answers' {text, answer_start}
    return ds
