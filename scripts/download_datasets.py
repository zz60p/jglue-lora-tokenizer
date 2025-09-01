import os
from datasets import load_dataset

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ensure_dir('data')
    # Download JNLI and JSQuAD splits from JGLUE
    for name, conf in [('JNLI','JNLI'), ('JSQuAD','JSQuAD')]:
        print(f"Loading {name} ...")
        ds = load_dataset('shunk031/JGLUE', conf)
        for split in ds.keys():
            n = len(ds[split])
            print(f"  split={split}, size={n}")
    print("Done.")

if __name__ == "__main__":
    main()
