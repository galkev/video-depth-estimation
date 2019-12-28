from data import MdffH5Data, SuperSlomoData, DatasetSubset


def MdffH5DataDbgSmall(*args, **kwargs):
    return DatasetSubset(MdffH5Data(*args, **kwargs), range(100, 100+30))


def SuperSlomoDataDbgSmall(*args, **kwargs):
    return DatasetSubset(SuperSlomoData(*args, **kwargs), range(0, 4))