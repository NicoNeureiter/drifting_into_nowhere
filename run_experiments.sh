#!/usr/bin/sh
python3 -m src.experiments.experiment_evaluation rrw 0 &
python3 -m src.experiments.experiment_evaluation rrw 200 &
python3 -m src.experiments.experiment_evaluation rrw inf &
python3 -m src.experiments.experiment_evaluation cdrw 0 &
python3 -m src.experiments.experiment_evaluation cdrw 200 &
python3 -m src.experiments.experiment_evaluation cdrw inf &
