#!/bin/bash

# base
python make.py --mode base --data ML100K --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode base --data ML100K --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode base --data ML1M --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode base --data ML1M --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode base --data Amazon --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode base --data Amazon --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode base --data Douban --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode base --data Douban --train_mode base --run test --num_experiments 4 --round 16


python make.py --mode base --data ML100K --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode base --data ML100K --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode base --data ML1M --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode base --data ML1M --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode base --data Amazon --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode base --data Amazon --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode base --data Douban --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode base --data Douban --train_mode mdr --run test --num_experiments 4 --round 16


python make.py --mode base --data ML100K --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode base --data ML100K --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode base --data ML1M --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode base --data ML1M --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode base --data Amazon --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode base --data Amazon --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode base --data Douban --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode base --data Douban --train_mode fdr --run test --num_experiments 4 --round 16


python make.py --mode base --data ML100K --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode base --data ML100K --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode base --data ML1M --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode base --data ML1M --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode base --data Amazon --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode base --data Amazon --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode base --data Douban --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode base --data Douban --train_mode assist --run test --num_experiments 4 --round 16



# match
python make.py --mode match --data ML100K --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode match --data ML100K --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode match --data ML1M --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode match --data ML1M --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode match --data Amazon --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode match --data Amazon --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode match --data Douban --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode match --data Douban --train_mode base --run test --num_experiments 4 --round 16


python make.py --mode match --data ML100K --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode match --data ML100K --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode match --data ML1M --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode match --data ML1M --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode match --data Amazon --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode match --data Amazon --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode match --data Douban --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode match --data Douban --train_mode mdr --run test --num_experiments 4 --round 16


python make.py --mode match --data ML100K --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode match --data ML100K --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode match --data ML1M --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode match --data ML1M --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode match --data Amazon --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode match --data Amazon --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode match --data Douban --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode match --data Douban --train_mode fdr --run test --num_experiments 4 --round 16


python make.py --mode match --data ML100K --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode match --data ML100K --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode match --data ML1M --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode match --data ML1M --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode match --data Amazon --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode match --data Amazon --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode match --data Douban --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode match --data Douban --train_mode assist --run test --num_experiments 4 --round 16



# cold_start
python make.py --mode cold_start --data ML100K --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML100K --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data ML1M --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML1M --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Amazon --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Amazon --train_mode base --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Douban --train_mode base --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Douban --train_mode base --run test --num_experiments 4 --round 16


python make.py --mode cold_start --data ML100K --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML100K --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data ML1M --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML1M --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Amazon --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Amazon --train_mode mdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Douban --train_mode mdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Douban --train_mode mdr --run test --num_experiments 4 --round 16


python make.py --mode cold_start --data ML100K --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML100K --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data ML1M --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML1M --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Amazon --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Amazon --train_mode fdr --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Douban --train_mode fdr --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Douban --train_mode fdr --run test --num_experiments 4 --round 16


python make.py --mode cold_start --data ML100K --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML100K --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data ML1M --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data ML1M --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Amazon --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Amazon --train_mode assist --run test --num_experiments 4 --round 16

python make.py --mode cold_start --data Douban --train_mode assist --run train --num_experiments 4 --round 16
python make.py --mode cold_start --data Douban --train_mode assist --run test --num_experiments 4 --round 16