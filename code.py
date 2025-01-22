import asyncio
import nest_asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMUsing():
    def __init__(self,model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dialogue_history = [
            "User: Hi!",
            "Bot: Hello! How can I assist you today?",
            "User: What is AI?",
            "Bot: AI stands for Artificial Intelligence, which enables machines to mimic human intelligence."
        ]
    def process(self,input_text):
        templet = "\n".join(self.dialogue_history + [f"User: {input_text}", "Bot:"])
        # 将文本转换为模型输入
        tokens = self.tokenizer.encode(input_text, return_tensors="pt")
        # 使用模型生成文本
        output_tokens = self.model.generate(
            tokens,
            max_length=120,  # 生成的最大长度
            num_return_sequences=1,  # 返回的序列数
            temperature=0.9,  # 控制生成的随机性
            top_p=0.9,  # nucleus采样的阈值
            do_sample=True  # 启用采样
        )
        # 解码生成的文本
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response_lines = output_text.split("\n")  # 按行分割
        bot_last_response = response_lines[-1].replace("Bot:", "").strip()  # 提取最后一行并去掉标记
        self.dialogue_history.append("User: {input_text}")
        self.dialogue_history.append("Bot: {bot_last_response}")
        return output_text
    
my_model =LLMUsing(model_name="prithivMLmods/Llama-Deepsync-1B")
# 激活嵌套事件循环 (仅在 Jupyter Notebook 环境中需要)
nest_asyncio.apply()
# your telegrame Token
TOKEN = "7967521995:AAGMxDkWxwI62j2HtX-Dd5TtZ4UYE8Aj2Ys"
# 定义处理消息的函数
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("你好！我是你的机器人。发送任何消息，我会回复你。")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text

    # 回复用户发送的消息
    await update.message.reply_text(f"{my_model.process(user_message)}")

async def main():
    # 创建 Application 对象
    application = Application.builder().token(TOKEN).build()

    # 添加命令处理器（/start）
    application.add_handler(CommandHandler("start", start))

    # 添加消息处理器
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # 启动机器人
    print("ChatBot start...")
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
 # type: ignore