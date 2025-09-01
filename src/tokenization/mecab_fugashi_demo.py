# Demo: MeCab/fugashi tokenization for Japanese
from fugashi import Tagger
tagger = Tagger()
text = '本日は晴天なり。LLMの分かち書きを試します。'
print([m.surface for m in tagger(text)])
