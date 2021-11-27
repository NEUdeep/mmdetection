
cd /workspace/mnt/storage/kanghaidong/new_video_project/mmdetection
pip install /workspace/mnt/storage/kanghaidong/khd-project/awesome_work_project/work_gitlab/fire-events/3th_pak/torch-1.6.0-cp36-cp36m-linux_x86_64.whl
pip install /workspace/mnt/storage/kanghaidong/khd-project/awesome_work_project/work_gitlab/fire-events/3th_pak/torchvision-0.7.0-cp36-cp36m-linux_x86_64.whl

# pip install timm
pip install dataclasses tensorboard
# pip uninstall timm
cd /workspace/mnt/storage/kanghaidong/cloud_project/apex
pip install -v --disable-pip-version-check --no-cache-dir ./

cd /workspace/mnt/storage/kanghaidong/new_video_project/mmdetection

pip install mmcv-full==1.3.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
pip install -r requirements/build.txt
pip install -v -e .

pip install timm==0.4.9
pip uninstall pycocotools
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"