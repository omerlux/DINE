python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 450 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB-20201018-170341 --single_gpu --gpu_device 4 --continue_train

python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 450 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB_TRAIN_MC_DIV --mc_eval 15 --mc_freq 10 --single_gpu --gpu_device 5

python main_dine.py --nhid 256 --ncell 16 --epochs 100 --save DINE --gpu_device 3

nohup python main_dine.py --data AWGN --ndim 1 --nhid 16 --ncell 8 --epochs 150 --N 1 --save DINE_AWGN_XDB --P  --gpu_device 3 &
nohup python main_dine.py --data AWGN --nhid 16 --ncell 8 --epochs 150 --N 1 --P 1 --save DINE_AWGN_0DB_dim --ndim  --gpu_device 3 &