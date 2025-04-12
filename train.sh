python -m lstm_bert_classification.services.pipeline --dataloader_workers 2 --device cuda --seed 42 --epochs 20 --learning_rate 5e-5 --weight_decay 0.01 --warmup_steps 50 --max_length 256 --max_seq_len 8 --pad_mask_id -100 --model vinai/phobert-base-v2 --train_batch_size 16 --val_batch_size 16 --test_batch_size 16 --train_file dialogue_dataset/train.json --val_file dialogue_dataset/val.json --test_file dialogue_dataset/test.json --output_dir ./bert_lstm_classification/models/classification --record_output_file output.json --early_stopping_patience 5 --early_stopping_threshold 0.001 --evaluate_on_accuracy