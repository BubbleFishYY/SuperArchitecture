# 载入 LLM 和 SamplingParams
from TD_Pipe import LLM, SamplingParams
# 推理数据以List[str]格式组织
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 设置采样参数
sampling_params = SamplingParams(temperature=0, top_p=1)
# 加载模型
llm = LLM(model="/fastdata/zhanghb/Mixtral-8x7B-v0.1",
          pipeline_parallel_size=2,
          tensor_parallel_size=2,
          seed=0,
          enforce_eager=True,
          )
# 执行推理
outputs = llm.generate(prompts, sampling_params)

# 输出推理结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")