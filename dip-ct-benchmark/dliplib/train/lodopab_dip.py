import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('validation', 2)

task_table = TaskTable()

# create the reconstructor
reconstructor = DeepImagePriorReconstructor(dataset.ray_trafo)

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr': [0.001, 0.0005, 0.0001],
                      'scales': [5],
                      'gamma': [0.0],
                      'channels': [(64,) * 5, (128,) * 5],
                      'skip_channels': [(0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 4),
                                        (0, 0, 0, 4, 4)],
                      'iterations': [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                      'loss_function': ['poisson']
                  })

results = task_table.run()

save_results_table(results, 'lodopab_dip--5-SCALES')

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('lodopab_dip')
