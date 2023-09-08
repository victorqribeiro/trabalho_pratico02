# run every model with both datasets and several epochs and batch_sizes combinations 
for script in 'AlexNet' 'LeNet' 'VGG'
do
  for dataset in 'cifar' 'fashion'
  do
    for epoch in 10 20 30 
    do
      for batch in 64 128 256
      do
        python3 $script.py $dataset $epoch $batch
      done
    done
  done
done
