pip install -r requirements.txt
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
cd src/lib/models/networks
rm DCNv2 -r
git clone https://github.com/lbin/DCNv2.git
cd DCNv2
git checkout -b pytorch_1.9 origin/pytorch_1.9
python setup.py build develop
cd ../../../..
python main.py ctdet --exp_id pascal_eca_18 --arch reseca_18 --dataset pascal --input_res 512 --num_epochs 140 --lr_step 45,60 --gpus 0 --batch_size 64 --num_workers 16