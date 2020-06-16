#python \
#train_hand.py --config="configs/rcnn2_egohand.yml" \
#        --hidden_size=32 \
#        --initial=0 \
#        --clip=10. \
#        --steps=3


python \
train_hand.py --config="configs/rcnn2_egohand.yml" \
        --hidden_size=32 \
        --initial=1 \
        --clip=10. \
        --steps=3


