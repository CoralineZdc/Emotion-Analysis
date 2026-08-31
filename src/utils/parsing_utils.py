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
    """
    Extract simulation parameters from the state dictionary path.

    Args:
        state_dict_path: Path to the state dictionary file.
    Returns:
        A dictionary containing the extracted simulation parameters.
    """
    dir_name = os.path.dirname(state_dict_path)
    dir_name = dir_name.split("/")[-1]  # Get the last part of the path
    params_list = dir_name.split("_")  # Split by underscores
    params = {}
    for param in params_list:
        for id_char in range(len(param)):
            if param[id_char].isdigit():
                break
        key = param[:id_char]
        value = param[id_char:]
        params[key] = value
    return params