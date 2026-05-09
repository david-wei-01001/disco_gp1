torchrun \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  --rdzv_id=my_job \
  DDP_run.py
