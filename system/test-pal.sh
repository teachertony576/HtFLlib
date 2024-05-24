#测试不同的公共数据集大小的影响
nohup python -u main.py -t 3  -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100-dir -m HtFE8 -fd 512 -did 0 -algo PAL_FL_notarget -pbs 30 -lam 10 -se 100 -mart 100 -gr 1000 > total-Cifar100-dir-HtFE8-fd=512-PAL_FL_notarget-Cifa10_public-1000-pbs_30.out 2>&1 &
nohup python -u main.py -t 3  -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100-dir -m HtFE8 -fd 512 -did 0 -algo PAL_FL -pbs 30 -lam 10 -se 100 -mart 100 -gr 1000 > total-Cifar100-dir-HtFE8-fd=512-PAL_FL-Cifa10_public-1000-pbs_30.out 2>&1 &
nohup python -u main.py -t 3  -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100-dir -m HtFE8 -fd 512 -did 2 -algo PAL_FL_notarget -pbs 20 -lam 10 -se 100 -mart 100 -gr 1000 > total-Cifar100-dir-HtFE8-fd=512-PAL_FL_notarget-Cifa10_public-1000-pbs_20.out 2>&1 &
nohup python -u main.py -t 3  -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100-dir -m HtFE8 -fd 512 -did 2 -algo PAL_FL -pbs 20 -lam 10 -se 100 -mart 100 -gr 1000 > total-Cifar100-dir-HtFE8-fd=512-PAL_FL-Cifa10_public-1000-pbs_20.out 2>&1 &

nohup python -u main.py   -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100-PathologicalSetting -m HtFE8 -fd 512 -did 3 -algo FedTGP_PAL -pbs 10 -lam 10 -se 100 -mart 30 -gr 1000 > total-Cifar100-PathologicalSetting-HtFE8-fd=512-FedTGP_PAL-Cifa10_public-1000-PED.out 2>&1 &
