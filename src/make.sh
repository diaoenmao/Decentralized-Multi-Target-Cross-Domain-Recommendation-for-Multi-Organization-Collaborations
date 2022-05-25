python make.py --file joint --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file alone --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file assist --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file ar --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file aw --data Douban --run train --round 16 --num_experiments 4 --round 16

python make.py --file ar-optim --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file optim-optim --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file match --data Douban --run train --round 16 --num_experiments 4 --round 16
python make.py --file pl --data Douban --run train --round 16 --num_experiments 4 --round 16

python make.py --file joint --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file alone --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file assist --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file ar --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file aw --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file ar-optim --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file optim-optim --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file match --data Douban --run test --round 16 --num_experiments 4 --round 16
python make.py --file pl --data Douban --run test --round 16 --num_experiments 4 --round 16