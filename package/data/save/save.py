"""Functions saving :
    - any ndarray in csv file 
    - problem meta parameters
"""
from csv import writer as csvwriter
from numpy import savetxt


def save_data(path: str, name: str, data):
    """Saves the data in csv file.

    Args:
        path (str): Path to the file to save folder.
        name (str): Name of the data to save. Will be the file name.
        data (ArrayLike): Data to save.
    """
    with open(f"{path}\\{name}.csv", "w", newline="", encoding="utf8") as f:
        savetxt(f, data, delimiter=";", fmt="%s")


def save_data_analysis(path: str, name: str, header, data):
    """Saves the analysis in csv file with header.

    Args:
        path (str): Path to the file to save folder.
        name (str): Name of the data to save. Will be the file name.
        header (list[str]): Header containing colnames.
        data (ArrayLike): Data to save.
    """
    with open(f"{path}\\{name}.csv", "w", newline="", encoding="utf8") as f:
        writer = csvwriter(f, delimiter=";")
        writer.writerow(header)
        savetxt(f, data, delimiter=";", fmt="%s")


def save_meta_data(
    path: str,
    exp_repeat: int,
    exp_cand: int,
    exp_pi: int,
    exp_criteria: int,
    low: float,
    high: float,
    epsilon: float,
    max_sum: float,
    fixed_sum: float,
    precision: int,
):
    """Saves the experiment meta data in csv file.

    Args:
        path (str): Path to the root of the experiment folder.
        exp_repeat (int): Number of dataset of the experiment.
        exp_cand (int): Number of candidates by dataset.
        exp_pi (int): Number of Prefential Information by dataset.
        exp_criteria (int): Number of criteria on which candidates are evaluated.
    """
    with open(f"{path}\\meta.csv", "w", newline="", encoding="utf8") as meta:
        writer = csvwriter(meta, delimiter=";")
        if precision == 0:
            writer.writerows(
                [
                    [exp_repeat],
                    [exp_cand],
                    [exp_pi],
                    [exp_criteria],
                    [int(low)],
                    [int(high)],
                    [int(epsilon)],
                    [int(max_sum)],
                    [int(fixed_sum)],
                    [precision],
                ]
            )
        else:
            writer.writerows(
                [
                    [exp_repeat],
                    [exp_cand],
                    [exp_pi],
                    [exp_criteria],
                    [low],
                    [high],
                    [epsilon],
                    [max_sum],
                    [fixed_sum],
                    [precision],
                ]
            )


def save_experiment_data_factory(path: str, file_name: str):
    """Cleans the data file and returns a generator object which receives
    the experiment data and append it.

    Args:
        path (str): Path to the file to save folder.
        file_name (str): Name of the data to save. Will be the file name.
    """
    open(f"{path}\\{file_name}", "w", newline="", encoding="utf8")

    def save_experiment_data():
        """Generator object which receives the experiment data and append it to the file."""
        with open(f"{path}\\{file_name}", "a", newline="", encoding="utf8") as file:
            writer = csvwriter(file, delimiter=";")
            count = 0
            while True:
                data = yield None
                count += 1
                writer.writerow(data)
                if count == 10:
                    file.flush()
                    count = 0

    f = save_experiment_data()
    f.send(None)
    return f
