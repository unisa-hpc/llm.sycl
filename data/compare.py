import numpy as np
import os


def get_files_with_pattern(path, has_word):
    files = os.listdir(path)
    return [f for f in files if f.find(has_word) != -1]

def find_diff(uut, gold):
    assert uut.shape == gold.shape
    assert len(uut.shape) == 1
    cnt = 0
    for i in range(uut.shape[0]):
        if abs(uut[i] - gold[i]) > 1e-1:
            print("\tFound a difference at index: ", i)
            print("\t\tuut: ", uut[i])
            print("\t\tgold: ", gold[i])
            cnt += 1
            if cnt > 100:
                print("Found more than 100 differences, stopping")
                break


if __name__ == '__main__':
    path = '/tmp/'
    golds = get_files_with_pattern(path, 'gold.npy')
    uuts = get_files_with_pattern(path, 'uut.npy')
    # print(golds)
    # print(uuts)

    for uut in uuts:
        case_uut = uut.split('_')[0]
        for gold in golds:
            case_gold = gold.split('_')[0]

            if case_uut == case_gold:
                print("Found a case: ", case_uut)
                uut_data = np.load(path + uut)
                gold_data = np.load(path + gold)
                # print(uut_data)
                # print(gold_data)
                match = np.allclose(uut_data, gold_data, atol=1e-1)
                print("comparison against gold passes: ", match)
                if not match:
                    find_diff(uut_data, gold_data)

                print('---')
