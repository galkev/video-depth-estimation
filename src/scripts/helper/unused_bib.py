from pathlib import Path
import os
import re
import json


def files_contain_text(files, text):
    for file in files:
        with open(file, "r") as f:
            if text in f.read():
                return True

    return False


def main():
    filt = "**/*.tex"
    path = "/home/kevin/Documents/master-thesis/thesis"

    files = list(Path(path).glob(filt))
    # print(files)

    with open(os.path.join(path, "bibliography.bib"), "r") as f:
        bibtex = f.read()

    keys = []
    keys_unused = []

    for bib_entry in re.findall(r"(@.*{)(.*),", bibtex):
        key = bib_entry[1].strip()
        keys.append(key)

    for key in keys:
        if not files_contain_text(files, key):
            keys_unused.append(key)

    print(len(keys_unused), "/", len(keys))
    print(json.dumps(keys_unused, indent=4))


if __name__ == "__main__":
    main()
