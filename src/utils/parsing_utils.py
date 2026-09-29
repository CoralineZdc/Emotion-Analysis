import os


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def parse_csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    strings_list = [x.strip() for x in value.split(",") if x.strip()]
    if strings_list == [""]:
        return []
    return strings_list


def parse_csv_bool_tuples(value):
    result = []
    for item in value.split(";"):
        item = item.strip("(").strip(")")
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid format for boolean tuple: {item}")
        parts = [bool(int(x.strip())) for x in parts]
        result.append(tuple(parts))
    return result


def get_simu_params(state_dict_path):
    """Extracts simulation parameters from the directory name of the state_dict_path."""
    dir_name = state_dict_path.parent.name
    params = {}
    for param in dir_name.split("_"):
        if "-" in param:
            key, value = param.split("-", 1)
            params[key] = value
        else:
            idx = next((i for i, c in enumerate(param) if c.isdigit()), len(param))
            key, value = param[:idx], param[idx:]
            if key:
                params[key] = value
    return params
