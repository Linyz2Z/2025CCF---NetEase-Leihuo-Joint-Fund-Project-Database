echo "Step 1: 进行aclue数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets aclue \
    --model_file ../code/configs/aclue_models.yaml \
    --final_test

echo "Step 2: 进行arc_c数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets arc_c \
    --model_file ../code/configs/arc_c_models.yaml \
    --final_test

echo "Step 3: 进行cmmlu数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets cmmlu \
    --model_file ../code/configs/cmmlu_models.yaml \
    --final_test

echo "Step 4: 进行hotpot_qa数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets hotpot_qa \
    --model_file ../code/configs/hotpot_qa_models.yaml \
    --final_test

echo "Step 5: 进行math数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets math \
    --model_file ../code/configs/math_models.yaml \
    --final_test

echo "Step 6: 进行mmlu数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets mmlu \
    --model_file ../code/configs/mmlu_models.yaml \
    --final_test

echo "Step 7: 进行squad数据集测试"
python ../code/test.py \
    --device cuda:2 \
    --batch_size 32 \
    --test_datasets squad \
    --model_file ../code/configs/squad_models.yaml \
    --final_test