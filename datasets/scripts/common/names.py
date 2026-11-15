"""Deterministic synthetic Chinese place-name strings."""
CHARS = tuple(dict.fromkeys('清和安乐明新宏瑞锦泽云景晨华康福泰阳祥雅宁兴嘉文远广德元中天湖海东南西北春夏秋冬江河山'))
SUFFIX = ('大厦', '大楼', '中心', '公寓', '商厦')


def make_name(index):
    count = len(CHARS)
    return CHARS[(index // count) % count] + CHARS[index % count] + SUFFIX[(index // (count * count)) % len(SUFFIX)]
