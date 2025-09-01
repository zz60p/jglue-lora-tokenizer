# Demo: SudachiPy tokenization with different modes
from sudachipy import dictionary, tokenizer as tknz
tokenizer_obj = dictionary.Dictionary().create()
text = '本日は晴天なり。LLMの分かち書きを試します。'
for mode in [tknz.Tokenizer.SplitMode.A, tknz.Tokenizer.SplitMode.B, tknz.Tokenizer.SplitMode.C]:
    morphemes = tokenizer_obj.tokenize(text, mode)
    print(mode, [m.surface() for m in morphemes])
