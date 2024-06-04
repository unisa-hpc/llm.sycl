import numpy as np
import os


def get_files_with_pattern(path, has_word):
    files = os.listdir(path)
    return [f for f in files if f.find(has_word) != -1]


def find_diff(uut, gold, len):
    max_diff = len / 10000000.0 * 0.1
    if max_diff > 2:
        print("WARNING: Max diff is too large (>1).")
    assert uut.shape == gold.shape
    assert uut.ndim == 1
    cnt = 0
    sum_diff = 0
    for i in range(uut.shape[0]):
        diff_abs = abs(uut[i] - gold[i])
        sum_diff += diff_abs
        if diff_abs > max_diff:
            print("\tFound a difference at index: ", i)
            print("\t\tuut: ", uut[i])
            print("\t\tgold: ", gold[i])
            cnt += 1
            if cnt > 100:
                print("Found more than 100 differences, stopping")
                return False
    if sum_diff < max_diff * 100:
        print("Sum of differences: ", sum_diff)
        return True
    else:
        print("Sum of differences is too large: ", sum_diff)
        return False

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
                    find_diff(uut_data, gold_data, len=gold_data.size)

                print('---')
