# Self Paced Contextual Reinforcement Learning
Implementation of the Self Paced Contextual Reinforcement Learning Experiments

## Installation

It is easiest to setup a virtual environment in order to install the required site-packages without modifying your global python installation. We are using Python3 (to be precise 3.5.2 on Ubuntu 16.04.5 LTS) and hence (assuming the code from this repository is in [DIR]), the following lines of code setup the virtualenv and install the required packages

```bash
cd [DIR]
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

If you want to run the experiments from the ''reacher'' or ''reacher-obstacle'' environment, you need will MuJoCo. If you have MuJoCo installed, be sure that you placed the corressponding binary and license key in the `~/.mujoco/` directory as described [here](https://github.com/openai/mujoco-py) (you may need to create the directory). This is necessary, because the mujoco-py package (which allows using MuJoCo from Python), relies on MuJoCo being located in this specific directory. If everything is setup, you need to run the following command (we assume that you still have the virtualenv activated):

```bash
pip3 install -r requirements_ext.txt
```

This will install OpenAI Gym and mujoco-py in the required versions.

## Usage

The experiments in the Gate Environment can be run with the following commands

```bash
python3 run_experiment.py --n_cores 10 --environment gate --setting precision --n_experiments 40 --algorithm sprl
python3 run_experiment.py --n_cores 10 --environment gate --setting precision --n_experiments 40 --algorithm creps
python3 run_experiment.py --n_cores 10 --environment gate --setting precision --n_experiments 40 --algorithm cmaes
python3 run_experiment.py --n_cores 10 --environment gate --setting precision --n_experiments 40 --algorithm goalgan
python3 run_experiment.py --n_cores 10 --environment gate --setting precision --n_experiments 40 --algorithm saggriac
python3 run_experiment.py --n_cores 10 --environment gate --setting global --n_experiments 40 --algorithm sprl
python3 run_experiment.py --n_cores 10 --environment gate --setting global --n_experiments 40 --algorithm creps
python3 run_experiment.py --n_cores 10 --environment gate --setting global --n_experiments 40 --algorithm goalgan
python3 run_experiment.py --n_cores 10 --environment gate --setting global --n_experiments 40 --algorithm saggriac
python3 visualize_results.py --n_cores 10 --environment gate --setting precision --add_cmaes --add_goalgan --add_saggriac
python3 visualize_results.py --n_cores 10 --environment gate --setting global --add_goalgan --add_saggriac
```

The first two commands create the experimental data in the "precision" setting for all algorithms. The next two commands do the same for the "global" setting. Finally, the results in both settings are visualized using the last command. Note that for the visualization, we recompute the performance and hence the visualization takes a bit of time when it is first run (not as much as the experiments however). However, the computed data is stored to disk and hence subsequent executions of the "visualize_results.py" script will render the data right away.

Allthough we create 10 subprocesses, our machine only had a quad-core processor (Core i7-7700), so it is not necessary to have 10 physical cores to run the script without problems. Note that you may nonetheless change the number of cores as desired. However, this also changes the seeds in the created subprocesses and hence will minimally alter the results.

To run the experiments in the other environments, you can use "--environment reacher-obstacle" or "--environment ball-in-a-cup". In this case, you do not need to set a "--setting" option. So to run the experiments in the modified reacher environment, you would e.g. run

```bash
python3 run_experiment.py --n_cores 10 --environment reacher-obstacle --n_experiments 40 --algorithm sprl
python3 run_experiment.py --n_cores 10 --environment reacher-obstacle --n_experiments 40 --algorithm creps
python3 run_experiment.py --n_cores 10 --environment reacher-obstacle --n_experiments 40 --algorithm cmaes
python3 run_experiment.py --n_cores 10 --environment reacher-obstacle --n_experiments 40 --algorithm goalgan
python3 run_experiment.py --n_cores 10 --environment reacher-obstacle --n_experiments 40 --algorithm saggriac
python3 visualize_results.py --n_cores 10 --environment reacher-obstacle --add_cmaes --add_goalgan --add_saggriac
```

Please note that this took multiple days in our lab setting, since the corressponding MuJoCo simulation is somewhat expensive.

You can see all the additional flags of the `run_experiment.py` and `visualize_results.py` scripts via

```bash
python3 run_experiment.py -h
python3 visualize_results.py -h
```
