"""Function reading results from an experiment"""
from csv import reader


def load_experiment_results(exp_path: str, file_location: str):
    """Loads results of explanation experiment from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
        file_location (str): Csv file name (contains its subfolder also).
    """
    with open(f"{exp_path}\\{file_location}", "r", newline="", encoding="utf8") as f:
        for x in reader(f, delimiter=";"):
            yield (int(x[0]), float(x[1]), int(x[2]))


def load_pairwise_length_data(exp_path: str, file_location: str):
    """Loads lengths of explanation experiment from csv file.

    Args:
        path (str): Path to the experiment's csv folder.
        file_location (str): Csv file name (contains its subfolder also).
    """
    with open(f"{exp_path}\\{file_location}", "r", newline="", encoding="utf8") as f:
        for x in reader(f, delimiter=";"):
            yield (int(x[0]), int(x[1]))
