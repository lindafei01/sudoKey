from transformers import AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型 - 使用 AutoModelForCausalLM 而不是 AutoModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 加载适配器模型
adapter_model = PeftModel.from_pretrained(base_model, "/home/yanlin/sudoKey/alignment-handbook/save/llama3-sudo-unlearned-qlora/checkpoint-6")

# 现在你可以使用 adapter_model 进行推理
# 例如：
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = adapter_model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))