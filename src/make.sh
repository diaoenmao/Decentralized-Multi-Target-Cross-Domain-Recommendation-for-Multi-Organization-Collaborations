for run in train test
do
#  for data in ML100K ML1M Amazon Douban
  for data in ML100K
  do
    python make.py --file joint --data $data --run $run --round 16 --num_experiments 4 --round 16
    python make.py --file alone --data $data --run $run --round 16 --num_experiments 4 --round 16
    python make.py --file assist --data $data --run $run --round 16 --num_experiments 4 --round 16
    python make.py --file ar --data $data --run $run --round 16 --num_experiments 4 --round 16
    python make.py --file ar-optim --data $data --run $run --round 16 --num_experiments 4 --round 16
  done
done

