

def is_list(l):
    return isinstance(l, list)


def list_len(l):
    return len(l) if is_list(l) else 1


def list_get(l, idx):
    if is_list(l):
        return l[idx]
    else:
        if idx == 0:
            return l
        else:
            raise Exception("Trying to access element from non list")