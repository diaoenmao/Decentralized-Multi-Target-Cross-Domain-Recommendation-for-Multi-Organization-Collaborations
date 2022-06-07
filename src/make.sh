num_experiments=1
round=16

for run in train test
do
#  for data in ML100K ML1M Amazon Douban
  for data in ML100K
  do
    python make.py --mode joint --data $data --run $run --num_experiments $num_experiments --round $round
    python make.py --mode alone --data $data --run $run --num_experiments $num_experiments --round $round
#    python make.py --mode assist --data $data --run $run --num_experiments $num_experiments --round $round
#    python make.py --mode ar --data $data --run $run --num_experiments $num_experiments --round $round
#    python make.py --mode ar-optim --data $data --run $run --num_experiments $num_experiments --round $round
  done
done

