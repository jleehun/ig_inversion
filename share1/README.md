# Evaluation of Input Attribs

```bash 
python generate_random_attrs.py

python run.py \
    --data-path /data/ImageNet1k/ \
    --model-name resnet50 \
    --attr-path results/attrs.npy \
    --measure morf \
    --ratio 0.1
```