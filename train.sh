export PYTHONPATH=./src
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/ds/documents/IIT-CDIP"  \
    --val-data="/ds/documents/IIT-CDIP"  \
    --warmup 10000 \
    --batch-size=$3 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=16 \
    --model ViT-B-16-plus-240 \
    --pretrained laion400m_e31 \
    --dataset-type custom \

# --test-run 1 \
# --train-data="/ds/documents/IIT-CDIP"  \
# --val-data="/ds/documents/IIT-CDIP"  \
# --train-data="/run/user/3841/gvfs/sftp:host=login1.pegasus.kl.dfki.de/ds/documents/IIT-CDIP"  \
# --val-data="/run/user/3841/gvfs/sftp:host=login1.pegasus.kl.dfki.de/ds/documents/IIT-CDIP"  \