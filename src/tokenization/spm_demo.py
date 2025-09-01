from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")
def spm_tokens(txt): return tok.convert_ids_to_tokens(tok(txt, add_special_tokens=False)["input_ids"])
if __name__=="__main__":
    t = "本日は晴天なり。生成モデルの分かち書きとサブワード分割を比較します。2024年のデータ。"
    print(spm_tokens(t))
