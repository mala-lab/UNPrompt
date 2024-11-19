
# for dataset in Amazon
# do
# for numprompts in 1 3 5 7 10 20 40 50 60
# do
# for edge in 0.0 0.1 0.2 0.3
# do
# for feat in 0.1 0.2 0.3
# do
# python train.py --dataset $dataset --numprompts $numprompts --edge_drop_prob $edge --feat_drop_prob $feat
# done
# done
# done
# done

python train.py --dataset Facebook --numprompts 1 --edge_drop_prob 0.1 --feat_drop_prob 0.3 --epochs 900

# python train.py --dataset Facebook --numprompts 1 --edge_drop_prob 0.2 --feat_drop_prob 0.3 --epochs 950

# python train.py --dataset Facebook --numprompts 1 --edge_drop_prob 0.3 --feat_drop_prob 0.3 --epochs 950

# python train.py --dataset Facebook --numprompts 5 --edge_drop_prob 0.3 --feat_drop_prob 0.3 --epochs 950

# python train.py --dataset Facebook --numprompts 1 --edge_drop_prob 0.1 --feat_drop_prob 0.0 --epochs 100

# python train.py --dataset Amazon --numprompts 5 --edge_drop_prob 0.1 --feat_drop_prob 0.2 --epochs 50

# python train.py --dataset Amazon --numprompts 1 --edge_drop_prob 0.2 --feat_drop_prob 0.2 --epochs 50

# python train.py --dataset Amazon --numprompts 1 --edge_drop_prob 0.2 --feat_drop_prob 0.3 --epochs 50

# python train.py --dataset Amazon --numprompts 1 --edge_drop_prob 0.3 --feat_drop_prob 0.1 --epochs 50

