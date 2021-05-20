import os
import pyarrow as pa
import pyarrow.dataset as ds
import webbrowser

from argparse import ArgumentParser

from utils import random_name

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the PARQUET dataset')
    args = parser.parse_args()

    dataset = ds.dataset(args.path)

    while True:
        user_input = input("Opinion ID: ")
        if user_input.lower() == 'stop':
            break

        if not user_input.isdigit():
            print("Wrong input. Must be an integer.")
            continue

        opid = int(user_input)
        scan_tasks = dataset.scan(filter=ds.field('opinion_id') == opid)
        batches = sum((list(s.execute()) for s in scan_tasks), [])
        # noinspection PyArgumentList
        t = pa.Table.from_batches(batches=batches)  # incorrect warning about this call

        if t.num_rows == 0:
            print("Opinion not found.")
            continue

        if t.num_rows > 1:
            print("Found more than one...")
            continue

        d = t.to_pydict()
        tmp = os.path.join('/tmp', random_name())
        with open(tmp, 'w') as f:
            f.write(d['html_with_citations'][0])
        print(f'Opinion written in {tmp}')
        webbrowser.open_new_tab(f'file://{tmp}')


if __name__ == '__main__':
    main()
