python main.py --dataset cora --weight_decay 1e-5 --dropout 0.4 --combine_result --store --lamda1 1
python main.py --dataset citeseer --weight_decay 1e-5 --dropout 0.4 --combine_result --store --lamda1 1
python main.py --dataset pubmed --weight_decay 5e-5 --tau 2 --dropout 0.3 --combine_result --store --lamda1 1 --alpha 3e-6
python main.py --dataset arxiv --weight_decay 5e-5 --tau 2 --dropout 0.2 --lamda1 1 --beta 0.1 --alpha 3e-6
python main.py --dataset twitch --weight_decay 3e-5 --tau 4 --dropout 0.1 --lamda1 1.5 --beta 0.5 --alpha 3e-6