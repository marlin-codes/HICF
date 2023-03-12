for margin in 0.6
do
  for num_layers in 4
  do
    for weight_decay in 0.001
    do
      for lr in 0.001
      do
        for num_neg in 3
        do
          for embedding_dim in 30 20 100
          do
          device_id=0
          CUDA_VISIBLE_DEVICES=${device_id} python run.py \
              --margin ${margin} \
              --num-layers ${num_layers} \
              --weight-decay ${weight_decay} \
              --log 1 \
              --lr ${lr} \
              --dataset yelp \
              --analysis dim_wise_analysis \
              --num_neg ${num_neg} \
              --embedding_dim ${embedding_dim}
        done
        done
      done
    done
  done
done