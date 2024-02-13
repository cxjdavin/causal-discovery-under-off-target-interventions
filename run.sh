# Setup
python3 -m venv myenv
source myenv/bin/activate
pip3 install --upgrade pip
pip3 install wheel networkx cvxpy causaldag tqdm p_tqdm matplotlib bnlearn
wget -r --no-parent --no-host-directories --cut-dirs=1 http://www.ics.uci.edu/\~eppstein/PADS/
cp pdag_patch.py myenv/lib/python3.9/site-packages/graphical_models/classes/dags/pdag.py

# Convert bnlearn graphs from BIF to nx
python3 bif_to_nx.py bif_bnlearn nx_bnlearn

# Run synthetic
python3 -m code.run_synthetic_experiments synthetic_config.json 10 r_hop 1 &
python3 -m code.run_synthetic_experiments synthetic_config.json 10 r_hop 2 &
python3 -m code.run_synthetic_experiments synthetic_config.json 10 decaying 0.5 &
python3 -m code.run_synthetic_experiments synthetic_config.json 10 decaying 0.9 &
python3 -m code.run_synthetic_experiments synthetic_config.json 10 fat_hand 0.5 &
python3 -m code.run_synthetic_experiments synthetic_config.json 10 fat_hand 0.9 &

wait

# Run bnlearn
python3 -m code.run_experiments config.json 10 r_hop 1 &
python3 -m code.run_experiments config.json 10 r_hop 2 &
python3 -m code.run_experiments config.json 10 decaying 0.5 &
python3 -m code.run_experiments config.json 10 decaying 0.9 &
python3 -m code.run_experiments config.json 10 fat_hand 0.5 &
python3 -m code.run_experiments config.json 10 fat_hand 0.9 &

