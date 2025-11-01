# model_name_or_path="Qwen/Qwen3-30B-A3B-Instruct-2507"
# model_name_or_path="Qwen/Qwen3-4B-Instruct-2507"

# vllm serve ${model_name_or_path} \
#     --port 8009 \
#     --served-model-name qwen \
#     --data-parallel-size 1 \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.8 \
#     --enable-log-requests

python ./examples/qwen3/runtime_demo.py