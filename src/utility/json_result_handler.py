import json
import os


class JsonResultReader:
    def __init__(self, file_name: str):
        self.__file_name = file_name

    def fetch_results(self) -> list:
        """ Fetch results from the json result file"""
        with open(self.__file_name, 'r') as json_file:
            results = json.load(json_file)
        return results


class JsonResultWriter:
    def __init__(self, file_name: str, overwrite: bool = False):
        if os.path.isfile(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise FileExistsError("Json result file is already existing!")
        self.__file_name = file_name

    @staticmethod
    def __convert_iterables(solution):
        """ Replaces iterable values with the length of the iterable"""
        solution_converted = dict()
        for k, v in solution.items():
            iterables = (dict, list, set, tuple)
            if type(v) in iterables:
                solution_converted[f"#{k}"] = len(v)
            else:
                solution_converted[k] = v
        return solution_converted

    def insert_result(self, solution: dict):
        """ Appends test result (solution) to the end of the json result file"""
        solution = self.__convert_iterables(solution)
        if os.path.isfile(self.__file_name):
            with open(self.__file_name, 'a+') as json_file:
                # remove last char in file before append - hacky but efficient
                json_file.seek(0, os.SEEK_END)
                json_file.seek(json_file.tell() - 1, os.SEEK_SET)
                json_file.truncate()
                json_file.write(',\n')
                json.dump(solution, json_file)
                json_file.write(']')
        else:
            results = list()
            results.append(solution)
            with open(self.__file_name, 'w') as json_file:
                json.dump(results, json_file)
        return

#
#
# if __name__ == '__main__':
#     sol1 = {
#         "number": 1.0,
#         "string": "sol1",
#         "tuple": (0, 1, 2),
#         "list": [0, 1, 2],
#         "dict": {"k1": "v1", "k2": "v2", "k3": "v3", },
#     }
#     sol2 = {
#         "number": 2,
#         "string": "sol2",
#         "tuple": (0, 1, 2),
#         "list": [0, 1, 2],
#         "dict": {"k1": "v1", "k2": "v2", "k3": "v3", },
#         "unequalamount": "ofparameters"
#     }
#     result_handler = JsonResultHandler(file_name=f"test.json", overwrite=True)
#     result_handler.insert_result(sol1)
#     result_handler.insert_result(sol2)
#     results = result_handler.fetch_results()
