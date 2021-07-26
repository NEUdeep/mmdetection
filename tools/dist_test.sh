# #!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch \
#     #  --format-only \
#     # --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco/openBrand_result_02" 
#     ${@:4}


#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
