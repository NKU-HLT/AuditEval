ENABLE_AUDIO_OUTPUT=False \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model ./AuditEval-llm/ckpt/auditeval-llm \
    --val_dataset ./AuditEval-llm/infer_batch.jsonl \
    --result_path ./AuditEval-llm/infer_batch_res.jsonl \
    --load_data_args false \
    --max_batch_size 4