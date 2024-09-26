import discord
from discord.ext import commands
from discord.commands import Option
import os
import dotenv
import torch
import huggingface_hub
import transformers
from transformers import pipeline

dotenv.load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)

bot.remove_command('help')

conversations = {}

huggingface_hub.login(os.getenv('HF_TOKEN'))
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} servers")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    
    mentions = message.mentions
    if not bot.user in mentions:
        return
    messages = []

    if message.author.id not in conversations:
        conversations[message.author.id] = [{"role": "system", "content": "You are a really cool chatbot who always responds with a chill vibe and uses emojis a lot."}]

    for message_ in conversations[message.author.id]:
        messages.append(message_)
    
    messages.append({"role": "user", "content": message.content})
    conversations[message.author.id].append({"role": "user", "content": message.content})

    reaction = "👍"
    await message.add_reaction(reaction)
    outputs = pipe(messages, max_new_tokens=256)

    await message.channel.send(outputs[0]["generated_text"][-1]['content'])

    conversations[message.author.id].append({"role": "assistant", "content": outputs[0]["generated_text"][-1]['content']})

bot.run(os.getenv("DISCORD_TOKEN"))
