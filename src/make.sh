num_experiments=4
round=16
resume_mode=0

for run in train test
do
  for data in ML100K ML1M Amazon Douban
  do
    python make.py --mode joint --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode alone --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode mdr --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode assist --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode match --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode info --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode pl --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode match-mdr --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode cs --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode cs-alone --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
    python make.py --mode cs-mdr --data $data --run $run --num_experiments $num_experiments --round $round --resume_mode $resume_mode
  done
done

