# Building starting models for full-waveform inversion using global optimization methods

This project aims to test a Particle swarm optimization algorithm (PSO) with an embedded elitism strategy for solving the 2D acoustic seismic inverse problem in an ideal scenario (synthetic data without noise). The project takes advantage of the DEAP and Devito frameworks for rapid prototyping and testing of ideas.

### Prerequisites

You need to install [DEAP](https://github.com/DEAP/deap) and [Devito](https://github.com/opesci/devito). Please see the corresponding web pages for installation instructions.

### Status

This is currently an experimental work in progress and is not ready. A problem with memory consumption needs to be fixed prior 
to the addition of [DEAP](https://github.com/DEAP/deap). The shot file from https://sesibahia-my.sharepoint.com/:u:/g/personal/oscar_ladino_fieb_org_br/ETC6Ocq1yJJElN-jS2fExXAB1BpT5rMO_S9hRd5ncl3VWA?e=Y9LIeg is required for testing.

## Authors

* **Oscar Mojica** - *Initial work* - [ofmla](https://github.com/ofmla)
