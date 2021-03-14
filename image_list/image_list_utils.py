
from tqdm import tqdm

if __name__ == '__main__':
    with open('./visda-2017-train.raw.txt') as fin, open('./visda-2017-train.txt', 'w') as fout:
        for line in tqdm(fin):
            fout.write(f'/data/VisDA-2017/train/{line}')
    print('Train finished!')

    with open('./visda-2017-validation.raw.txt') as fin, open('./visda-2017-validation.txt', 'w') as fout:
        for line in tqdm(fin):
            fout.write(f'/data/VisDA-2017/validation/{line}')
    print('Validation finished!')
