CUDA_VISIBLE_DEVICES=1
python main.py \
  --decay=5e-5 \
  --lr=0.001 \
  --layer=4 \
  --seed=1234 \
  --dataset="Amazon-Book" \
  --topks="[5,10,20,50]" \
  --recdim=50