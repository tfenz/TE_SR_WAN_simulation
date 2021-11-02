# Traffic Engineering with Joint Link Weight and Segment Optimization

## Overview 
The code has been the basis for computational evaluations within the publication '**Traffic Engineering with Joint Link Weight and Segment Optimization**'.
This repository contains all implemented algorithms, traffic and topology generators. Additionally, we provide the raw results (JSON), and the plotting script used for Fig. (3)-(5).

## Dependencies and Requirements
We implemented the proposed algorithms in [Python (3.7.10)](https://www.python.org/downloads/release/python-3710/) leveraging the library [NetworkX (2.5.1)](https://networkx.github.io/documentation/networkx-2.4/) and [NetworKit (8.1)](https://github.com/networkit/networkit). 
To solve the ILP we used [Gurobi (9.1.2)](https://www.gurobi.com/downloads/gurobi-software/).  
We used [conda (4.8.2)](https://anaconda.org/anaconda/beautifulsoup4/files?version=4.8.2) as a package manager. See the conda [environment.yml](environment.yml) for further details of packages used in this repository.

The code is tested on Ubuntu and MacOS. The python library *NetworKit* does not support Microsoft Windows in version 8.1.
The host machine in our evaluations was running Ubuntu 18.04.5 LTS.

## Structure

| Directory                           | Description                                                                     |
|-------------------------------------|---------------------------------------------------------------------------------|
| **[data/](data)**                   | Target directory for real-world traffic/topologies from SNDLib and TopologyZoo  |
| **[results_paper/](results_paper)** | Raw result data (json) used in the evaluations shown in the paper               |
| **[out/](src)**                     | To store json results and plots                                                 |
| **[src/](src)**                     | Source root containing *main.py* and plot_results.py                            |
| **[src/algorithm/](src/algorithm)** | WAN Routing algorithms (link weight and/or segment optimizations)               |
| **[src/topology/](src/topology)**   | Topology provider (reads/prepares available real-world topology data)           |
| **[src/demand/](src/demand)**       | Reader for real world traffic data and synthetic traffic generator              |
| **[src/utility/](src/utility)**     | Globally shared statics/consts and helper classes (e.g. JSON reader/writer)     |

## Prerequisites
### Conda
We use Conda as package manager and provide an environment.yml defining the conda environment used in the evaluations.
For details go to: [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

### Gurobi
We used Gurobi to solve linear problems. To reproduce the results a licence is required (academic licences are freely available here: 
[info](https://www.gurobi.com/academia/academic-program-and-licenses/)). 
Download and install the Gurobi Optimizer (9.1.2) from [download](https://www.gurobi.com/downloads/).

## Real-World Data
To tune our experiments interestingly, we use real world data for both - topologies and demands from [SNDLib](http://sndlib.zib.de/home.action) and [TopologyZoo](http://www.topology-zoo.org/dataset.html).

Overview of real-world data usage
* Fig. 3 (all topologies): Topology data from SNDLib and TopologyZoo.
* Fig. 4 (all algorithms): Topology data from SNDLib.
* Fig. 5 (real demands): Topology and traffic data from SNDLib.

### SNDLib Data
We use traffic and topology data from SNDLib, which we redistribute under the [ZIB ACADEMIC LICENSE](data/LICENSE_SNDLib).
The data is stored in the directory **[data/](data)**.

### TopologyZoo Data
Additionally, we use the topology data available from [TopologyZoo](http://www.topology-zoo.org/dataset.html).

**Note:** The data from topology zoo is **NOT** included in the repository and must be manually added:
1. Download the whole dataset: [Download](http://www.topology-zoo.org/files/archive.zip)
2. Unzip the data
3. Save the *.graphml files in the directory [data/topologies/topology_zoo](data/topologies/topology_zoo/))

## Install Python & Dependencies
Create a conda environment and install all python dependencies using the provided environment.yml file:
```bash
conda env create -f environment.yml
```
The created environment is named 'wan_sr', activate with:
```bash
conda activate wan_sr
```

## Run Tests
Navigate to source code root:
```bash
cd ./src
```

### Start 
Run evaluation with:
```bash
python3 main.py
```

### Output
The results are stored in a JSON file located in **[out/](src)** after running the main.py script.
*Note: In the directory **[results_paper/](results_paper)** we provide the raw results obtained during our evaluations which we used in the publication.*

## Plot Results
Create Plots from provided raw result data 
```bash
python3 plot_results.py [optional <data-dir> containing json result data]
```
*Note: By default, the script plots the raw result data used in Fig.3-5 in the paper. To plot the data created by running the main.py script, you can pass the directory containing the json files as parameter to the plotting script. E.g.:* 
```bash
python3 plot_results.py "../out/"
```

<p float="left">
  <a href="https://www.acm.org/publications/policies/artifact-review-and-badging-current"><img src="/artifacts_available_v1_1.png" width="150" /></a>
  <a href="https://www.acm.org/publications/policies/artifact-review-and-badging-current"><img src="/artifacts_evaluated_functional_v1_1.png" width="150" /></a>
  <a href="https://www.acm.org/publications/policies/artifact-review-and-badging-current"><img src="/artifacts_evaluated_reusable_v1_1.png" width="150" /></a>
</p>

## Contact
*[Contact Me](mailto:thomas.fenz@univie.ac.at)* or visit [University of Vienna | Communication Technologies](https://ct.cs.univie.ac.at/) for more infos.

*This project is licensed under the [MIT License](LICENSE)*.

