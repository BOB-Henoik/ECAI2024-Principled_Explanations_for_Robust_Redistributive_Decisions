from os import makedirs
from os.path import exists as pathexists
from collections import Counter
from itertools import tee, chain
from typing import List, Any, Dict
from statistics import quantiles, fmean, StatisticsError
from package.data.save import save_data_analysis, save_data
from package.data.load import (
    load_meta_data,
    load_experiment_results,
)


def lambda_factory(i: int):
    """Auxilary function returning the filter to apply to data according
    to the future row index of the file.

    Args:
        i (int): Value of the index
    """
    if i < 4:
        return lambda x: x[0] != 0
    if i < 8:
        return lambda x: x[0] == -2
    if i < 12:
        return lambda x: x[0] == -1
    return lambda x: x[0] > 0


def function_selector(i):
    """Auxilary function returning the method to be applyed according to index value.

    Args:
        i (int): Value of the index
    """
    if i % 4 == 0:
        return fmean
    if i % 4 == 1:
        return min
    if i % 4 == 2:
        return quantiles
    return max


def experiment_comparison(exp_path: str, file_name: str, include_timeout: bool = False):
    """Builds three files about the explanation function computation:
        - raw lengths effectives
        - distribution of computation time (overall, when Farkas failed,
        when the method failed, when it worked overall and detailed by number of PI used)
        - distribution of explanation lengths (overall, when Farkas failed,
        when the method failed, when it worked overall and detailed by number of PI used)

    Args:
        exp_path (str): Path to the experiment's csv folder.
        file_name (str): Explanation function file path and name.
    """
    nb_exp, nb_pi = [
        int(k) for i, k in enumerate(load_meta_data(exp_path)) if i in [0, 2]
    ]

    readers = tee(
        chain.from_iterable(
            load_experiment_results(f"{exp_path}\\{fold}", file_name)
            for fold in range(nb_exp)
        ),
        21 + 9 * nb_pi,
    )

    counts: List[Counter] = [Counter() for _ in range(nb_pi + 1)]
    time_computations: List[Any] = list(0 for _ in range(16 + nb_pi * 4))
    length_computations: List[Any] = list(0 for _ in range(4 + nb_pi * 4))

    counts[0].update((x for (x, _, _) in readers[0]))
    for i in range(1, nb_pi + 1):
        counts[i].update((x for (x, _, z) in readers[i] if z == i))

    it = 1 + nb_pi
    for i in range(16):
        try:
            time_computations[i] = function_selector(i)(
                (y for (x, y, _) in readers[it + i] if include_timeout or x > -3)
            )
        except StatisticsError:
            time_computations[i] = "NaN"
        except ValueError:
            time_computations[i] = "NaN"

    it += 16
    for i in range(nb_pi):
        for j in range(4):
            try:
                time_computations[16 + i * 4 + j] = function_selector(j)(
                    (y for (_, y, z) in readers[it + i * 4 + j] if z == i + 1)
                )
            except StatisticsError:
                time_computations[16 + i * 4 + j] = "NaN"
            except ValueError:
                time_computations[16 + i * 4 + j] = "NaN"

    # Length computations
    it += 4 * nb_pi
    for i in range(4):
        try:
            length_computations[i] = function_selector(i)(
                (x for (x, _, _) in readers[it + i] if x > 0)
            )
        except StatisticsError:
            length_computations[i] = "NaN"
        except ValueError:
            length_computations[i] = "NaN"

    it += 4
    for i in range(nb_pi):
        for j in range(4):
            try:
                length_computations[4 + i * 4 + j] = function_selector(j)(
                    (x for (x, _, z) in readers[it + i * 4 + j] if z == i + 1)
                )
            except StatisticsError:
                length_computations[4 + i * 4 + j] = "NaN"
            except ValueError:
                length_computations[4 + i * 4 + j] = "NaN"

    # Path creation
    folder = file_name.split("\\")[0]
    name = file_name.split("\\")[-1][:-4]
    file_path = f"{exp_path}\\Analysis\\Individual\\{folder}\\{name}"
    if not pathexists(file_path):
        makedirs(file_path)

    # Save time distribution
    header = ["Name", "#Found", "Mean", "Min", "Q1", "Median", "Q3", "Max"]
    time_data = []
    row_starters_time = [
        [
            "All",
            sum(counts[0].values()) - counts[0][-3]
            if not include_timeout
            else sum(counts[0].values()),
        ],
        ["No Farkas", counts[0][-2]],
        ["No explanation", counts[0][-1]],
        [
            "Found",
            sum(counts[0].values()) - counts[0][-3] - counts[0][-2] - counts[0][-1],
        ],
    ]
    pi_starters = [[f"{i} PI", sum(counts[i].values())] for i in range(1, nb_pi + 1)]

    deserealize_time = min_quartiles_max_deserialize_factory(time_computations)

    for i in range(4):
        time_data.append(row_starters_time[i] + [*deserealize_time(4 * i)])
    for i in range(nb_pi):
        time_data.append([*pi_starters[i], *deserealize_time(16 + 4 * i)])

    save_data_analysis(file_path, "time_distribution", header, time_data)

    # Save length counts
    save_data(file_path, "length_data", list(sorted(counts[0].items())))

    # Save length distributions
    deserealize_len = min_quartiles_max_deserialize_factory(length_computations)
    len_data = [
        [
            "Found",
            sum(counts[0].values()) - counts[0][-3] - counts[0][-2] - counts[0][-1],
            *deserealize_len(0),
        ]
    ]
    for i in range(nb_pi):
        len_data.append([*pi_starters[i], *deserealize_len(4 + 4 * i)])
    save_data_analysis(file_path, "length_distribution", header, len_data)


def min_quartiles_max_deserialize_factory(computations):
    """Builds the min_quartiles_max_deserialize function

    Args:
        computations (List): List containing the computations of data dispersion
    """

    def min_quartiles_max_deserialize(index: int):
        """Return in the following order :
            - the mean
            - the minimum
            - q1
            - the median
            - q3
            - the maximum
        starting from the given index.

        Args:
            index (int): Value of starting index.
        """
        quartiles = computations[index + 2]
        return (
            computations[index],
            computations[index + 1],
            quartiles[0] if quartiles != "NaN" else "NaN",
            quartiles[1] if quartiles != "NaN" else "NaN",
            quartiles[2] if quartiles != "NaN" else "NaN",
            computations[index + 3],
        )

    return min_quartiles_max_deserialize


class ExplLength:
    """Contains the Counter of the difference between the lengths of method 1
    and method 2, grouped by lengths of method 1."""

    def __init__(self) -> None:
        self.d: Dict[int, Counter] = {}

    def update_length(self, length: int, value: int):
        if length not in self.d:
            self.d[length] = Counter((value,))
        else:
            self.d[length].update((value,))

    def sort_by_length(self):
        self.d = dict(sorted(self.d.items()))

    def get_keys(self):
        return self.d.keys()

    def get_lengths(self):
        return self.d


def pairwise_generator(exp_path, nb_exp, file_1_name, file_2_name):
    for fold in range(nb_exp):
        for (x1, y1, _), (x2, y2, _) in zip(
            load_experiment_results(f"{exp_path}\\{fold}", file_1_name),
            load_experiment_results(f"{exp_path}\\{fold}", file_2_name),
        ):
            if x1 > 0 and x2 > 0:
                yield (y1 - y2) / y1 * 100, x1, y1, x2, y2


def pairwise_generator_timedout(exp_path, nb_exp, file_1_name, file_2_name):
    for fold in range(nb_exp):
        for (x1, y1, _), (x2, y2, _) in zip(
            load_experiment_results(f"{exp_path}\\{fold}", file_1_name),
            load_experiment_results(f"{exp_path}\\{fold}", file_2_name),
        ):
            if x1 == -3 and x2 > 0:
                yield (y1 - y2) / y1 * 100, x2, y1, x2, y2
            elif x2 == -3 and x1 > 0:
                yield (y1 - y2) / y1 * 100, x1, y1, x1, y2
            elif x1 > 0 and x2 > 0:
                yield (y1 - y2) / y1 * 100, x1, y1, x2, y2


def experiment_pairwise_comparison(
    exp_path: str, file_1_name: str, file_2_name: str, generator=pairwise_generator
):
    nb_exp = int(load_meta_data(exp_path)[0])

    reader = generator(exp_path, nb_exp, file_1_name, file_2_name)

    expls: Counter = Counter()
    expls_by_len: ExplLength = ExplLength()

    for _, x1, y1, x2, _ in reader:
        expls.update((x1 - x2,))
        expls_by_len.update_length(x1, x1 - x2)
        # if x1 > x2:
        #     print(f"Alerte : {(x1, y1)} in method 1")

    expls_by_len.sort_by_length()

    observed_1_lengths = expls_by_len.get_keys()

    readers = tee(
        generator(exp_path, nb_exp, file_1_name, file_2_name),
        8 + 8 * len(observed_1_lengths),
    )
    nb_readers_time = 4 + 4 * len(observed_1_lengths)
    time_computations: List[Any] = list(
        0 for _ in range(4 + len(observed_1_lengths) * 4)
    )
    len_computations: List[Any] = list(
        0 for _ in range(4 + len(observed_1_lengths) * 4)
    )

    for i in range(4):
        try:
            time_computations[i] = function_selector(i)(
                (t for t, _, _, _, _ in readers[i])
            )
        except StatisticsError:
            time_computations[i] = "NaN"
        except ValueError:
            time_computations[i] = "NaN"
        try:
            len_computations[i] = function_selector(i)(
                (x1 - x2 for _, x1, _, x2, _ in readers[nb_readers_time + i])
            )
        except StatisticsError:
            len_computations[i] = "NaN"
        except ValueError:
            len_computations[i] = "NaN"

    for i, length in enumerate(observed_1_lengths):
        for j in range(4):
            try:
                time_computations[4 + i * 4 + j] = function_selector(j)(
                    (t for t, x1, _, _, _ in readers[4 + i * 4 + j] if x1 == length)
                )
            except StatisticsError:
                time_computations[4 + i * 4 + j] = "NaN"
            except ValueError:
                time_computations[4 + i * 4 + j] = "NaN"
            try:
                len_computations[4 + i * 4 + j] = function_selector(j)(
                    (
                        x1 - x2
                        for _, x1, _, x2, _ in readers[nb_readers_time + 4 + i * 4 + j]
                        if x1 == length
                    )
                )
            except StatisticsError:
                len_computations[4 + i * 4 + j] = "NaN"
            except ValueError:
                len_computations[4 + i * 4 + j] = "NaN"

    # Path creation
    folder = file_1_name.split("\\")[0]
    file_name = (
        file_1_name.split("\\")[-1][:-4] + "_vs_" + file_2_name.split("\\")[-1][:-4]
    )
    file_path = f"{exp_path}\\Analysis\\Pairwise\\{folder}\\{file_name}"
    if not pathexists(file_path):
        makedirs(file_path)

    # Save time distribution
    len_header = [
        "Length Method 1",
        "#Found",
        r"%equal",
        r"%better",
        r"%worst",
        "Mean",
        "Min",
        "Q1",
        "Median",
        "Q3",
        "Max",
    ]

    deserealize_len = min_quartiles_max_deserialize_factory(len_computations)
    lengths_counter = expls_by_len.get_lengths()

    len_data = [
        [
            "All",
            *percentage_equal_better_worse_deserialize(expls),
            *deserealize_len(0),
        ]
    ]
    for i, length in enumerate(observed_1_lengths):
        len_data.append(
            [
                i,
                *percentage_equal_better_worse_deserialize(lengths_counter[length]),
                *deserealize_len(4 + 4 * i),
            ]
        )
    save_data_analysis(file_path, "length_distribution", len_header, len_data)

    time_header = ["Name", "#Found", "Mean", "Min", "Q1", "Median", "Q3", "Max"]
    deserealize_time = min_quartiles_max_deserialize_factory(time_computations)
    time_data = [["All", sum(expls.values()), *deserealize_time(0)]]

    for i, length in enumerate(observed_1_lengths):
        time_data.append(
            [i, sum(lengths_counter[length].values()), *deserealize_time(4 + 4 * i)]
        )

    save_data_analysis(file_path, "time_distribution", time_header, time_data)

    save_data(file_path, "length_data", list(sorted(expls.items())))

    if not pathexists(f"{file_path}\\length_data"):
        makedirs(f"{file_path}\\length_data")
    for i in observed_1_lengths:
        save_data(
            f"{file_path}\\length_data",
            str(i),
            list(sorted(lengths_counter[i].items())),
        )


def percentage_equal_better_worse_deserialize(counter: Counter):
    """Return in the following order :
        - the total number of values
        - the ratio of explanation lengths where the 2 methods are equal
        - the ratio of explanation lengths where methods 1 is better than method 2
        - the ratio of explanation lengths where method 1 is worst than method 2

    Args:
        counter (Counter): Counter of length difference values.
    """
    total_values = sum(counter.values())
    if total_values > 0:
        return (
            total_values,
            counter[0] / total_values * 100,
            sum(v for i, v in counter.items() if i > 0) / total_values * 100,
            sum(v for i, v in counter.items() if i < 0) / total_values * 100,
        )
    return 0, "NaN", "NaN", "NaN"
