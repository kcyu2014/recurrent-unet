#python \
#train_hand.py --config="configs/rcnn2_egohand.yml" \
#        --hidden_size=32 \
#        --initial=0 \
#        --clip=10. \
#        --steps=3

#
#python \
#train_hand.py --config="configs/runet_eythhand.yml" \
#        --hidden_size=32 \
#        --initial=1 \
#        --clip=10. \
#        --steps=3


python train_hand.py --config=configs/dataset/eythtest.yml \
    --model=runet --gate=3 --initial=1 \
    --scale_weight=0.4 --hidden_size=256 \
    --prefix='test'
