Installation

conda env create -f environment.yml
conda activate ant_sim
pip install torch --index-url https://download.pytorch.org/whl/cu130   # cu126 if nvidia-smi shows CUDA < 13.0
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Collect data from antsim
rm -rf runs/dataset_v0
bash run.sh

Simple sim run
python run.py --random_food --seed 0 --n_patches 2 --n_ants 50

Buid Dataset
python -m dataset.build_dataset
python -m dataset.synth

Check model
python -m model
python -m model --overfit

train model
python -m model.train --run_name smoke --steps 500 --eval_every 100
python -m model.train --run_name tf_wd01
python -m model.train --run_name tf_wd1 --weight_decay 1.0
python -m model.train --run_name tf_wd0 --weight_decay 0
python -m model.train --run_name mlp    --arch mlp

watch run during training
tail -f outputs/tf_wd01/metrics.jsonl

Look at the results
python -m model.plot outputs/tf_wd01
python -m model.plot outputs/tf_wd01 outputs/tf_wd1 outputs/tf_wd0 outputs/mlp --out compare.png --logx
python -m model.evaluate outputs/tf_wd01
python -m model.evaluate outputs/tf_wd01 --splits test
