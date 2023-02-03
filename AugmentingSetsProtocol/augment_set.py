import concurrent.futures as cf
import sys
import os

MAX_WORKERS = os.cpu_count() + 4


def augment_string(s: str, t: int):
    def augment_string_recursive(t1, n=0, prev=""):
        if n == len(s):
            if t1 == 0:
                res.add(prev)
                return
            augment_string_recursive(t1 - 1, n, prev + '*')
        else:
            augment_string_recursive(t1, n + 1, prev + s[n])
            if t1 > 0:
                augment_string_recursive(t1 - 1, n, prev + '*')
                augment_string_recursive(t1 - 1, n + 1, prev + '*')

    res = set()
    augment_string_recursive(t)
    return res


def print_usage():
    print("Usage:   python3 augment_set.py threshold input_file [-o output_file]")
    sys.exit(1)


def main():
    if len(sys.argv) == 2:
        augment_set(int(sys.argv[1]), sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[3] == '-o':
        augment_set(int(sys.argv[1]), sys.argv[2], sys.argv[4])
    else:
        print_usage()


def augment_set(t, in_file, out_file=''):
    if not out_file:
        out_file = "augmented_set.txt"
    res = set()
    with open(in_file, 'r') as f, cf.ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures_set = set()
        for line in f.readlines():
            future = executor.submit(augment_string, line.strip(), t)
            futures_set.add(future)
        for future in cf.as_completed(futures_set):
            res.update(future.result())
    with open(out_file, 'w') as f:
        for s in res:
            f.write(s + '\n')


if __name__ == '__main__':
    main()
