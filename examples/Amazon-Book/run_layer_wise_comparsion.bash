for margin in 0.5
do
  for num_layers in 2 3 4 6 7
  do
    for weight_decay in 0.001
    do
      for lr in 0.001
      do
        for num_neg in 12
        do
        device_id=1
        CUDA_VISIBLE_DEVICES=${device_id} python run.py \
              --margin ${margin} \
              --num-layers ${num_layers} \
              --weight-decay ${weight_decay} \
              --log 1 \
              --lr ${lr} \
              --dataset Amazon-Book \
              --analysis layer_wise_analysis \
              --num_neg ${num_neg}
      done
      done
    done
  done
done