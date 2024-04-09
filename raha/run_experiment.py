import os
import json
import yaml
import time
from tqdm import tqdm
import argparse
import itertools
from pathlib import Path

import dotenv
import mimir_dataset as dataset
import correction
import helpers

def run_baran(c: dict):
    """
    Runs baran with a configuration as specified in c.
    @param c: a dictionary that contains all parameters the experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c["run"] + 1  # RENUVER dataset-versions are 1-indexed
    start_time = time.time()

    try:
        data = dataset.Dataset(c['dataset'], c['error_fraction'], version, c['error_class'], c['n_rows'])
        data.detected_cells = data.get_errors_dictionary()

        app = correction.Correction()
        app.VERBOSE = False
        correction_dictionary = app.run(data)
        end_time = time.time()

        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:] 

        return {
                "status": 1,
                "result": {
                    "precision": p, "recall": r, "f1": f, "runtime": end_time - start_time
                    },
                "config": c,
                }
    except Exception as e:
        return {"status": 0,
                "result": str(e),
                "config": c}


def combine_configs(ranges: dict, config: dict, runs: int):
    """
    Calculate all possible configurations from combining ranges with the
    static config.
    ranges: dict of lists, where the key identifier the config-parameter,
    and the list contains possible values for that parameter.
    config: dict containing all parameters needed to run Baran. Keys
    contained in ranges get overwritten.
    runs: integer indicating how often a measurement should be repeated with
    one combination.
    @return: list of config-dicts
    """
    config_combinations = []
    ranges = {**ranges, "run": list(range(runs))}

    range_combinations = itertools.product(*list(ranges.values()))
    for c in range_combinations:
        combination = {}
        for i, key_range in enumerate(ranges.keys()):
            combination[key_range] = c[i]
        config_combinations.append(combination)

    configs = [
        {**config, **range_config} for range_config in config_combinations
    ]

    return configs


def load_dedicated_experiments(saved_config: str):
    config_path = Path('../infrastructure/saved-experiment-configs') / f'{saved_config}.yaml'
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f)

    if config.get('ranges_baran') is None:
        baran_configs = []
    else:
        baran_configs = combine_configs(ranges=config['ranges_baran'],
                                        config=config['config_baran'],
                                        runs=config['runs'])
    
    if config.get('ranges_renuver') is None:
        renuver_configs = []
    else:
        renuver_configs = combine_configs(ranges=config['ranges_renuver'],
                                        config=config['config_renuver'],
                                        runs=config['runs'])

    if config.get('ranges_openml') is None:
        openml_configs = []
    else:
        openml_configs = combine_configs(ranges=config['ranges_openml'],
                                        config=config['config_openml'],
                                        runs=config['runs'])

    print(f'Successfully loaded experiment configuration from {saved_config}.')
    configs = [*baran_configs, *renuver_configs, *openml_configs]
    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start experiment either on a k8s cluster, or on a dedicated machine.')
    parser.add_argument('--dedicated', action='store_true', help='Start experiment on a dedicated machine.', default=False)
    parser.add_argument('--saved-config', type=str, help='Name of the saved experiment configuration.', default=None)

    args = parser.parse_args()

    if args.dedicated:
        if args.saved_config is None:
            raise ValueError('No saved config specified to load experiment from.')

        configs = load_dedicated_experiments(args.saved_config)
        print(f'Start measurement for experiment {args.saved_config}.')

        results_path = Path('measurements/')
        results_path.mkdir(parents=True, exist_ok=True)
        for i, config in enumerate(tqdm(configs)):
            experiment_id = f'{args.saved_config}-{i}'
            experiment_id = experiment_id.replace('_', '-')
            result = run_baran(config)
            with open(results_path / f"{experiment_id}.json", 'wt') as f:
                json.dump(result, f)
        
    else:
        # To use the setup with Kubernetes, read the datasets name from an environment variable.
        config = json.loads(os.environ.get('CONFIG', '{}'))

        if len(config) == 0:
            raise ValueError('No config specified')

        experiment_id = os.environ.get('EXPERIMENT_ID')
        save_path = f"/measurements/{experiment_id}.json"

        result = run_baran(config)

        with open(save_path, 'wt') as f:
            json.dump(result, f)
        print('Running on k8s cluster')
