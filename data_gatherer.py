import pandas as pd


def create_series_from_ratios(phase_grains_dict):
    data = []
    for phase in phase_grains_dict:
        for grain in phase_grains_dict[phase]:
            row = [ratio for ratio in grain.calculatedRatiosDict.values()]
            data.append(row)

    results = pd.DataFrame(data, columns=grain.calculatedRatiosDict.keys())
    return results


def save_results_to_excel_file(phase_grains_dict, file_name="grain_results.xlsx"):
    data = []
    for phase in phase_grains_dict:
        for grain in phase_grains_dict[phase]:
            row = [ratio for ratio in grain.calculatedRatiosDict.values()]
            data.append(row)

    results = pd.DataFrame(data, columns=grain.calculatedRatiosDict.keys())
    print(f"Saving results to file ./Results/{file_name}")
    results.to_excel(f"./Results/{file_name}")
