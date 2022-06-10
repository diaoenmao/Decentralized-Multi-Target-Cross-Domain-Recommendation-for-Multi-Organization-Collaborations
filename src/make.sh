num_experiments=4
round=16

for run in train test
do
#  for data in ML100K ML1M Amazon Douban
  for data in ML100K ML1M
  do
    python make.py --mode joint --data $data --run $run --num_experiments $num_experiments --round $round
    python make.py --mode alone --data $data --run $run --num_experiments $num_experiments --round $round
    python make.py --mode assist --data $data --run $run --num_experiments $num_experiments --round $round
  done
done

