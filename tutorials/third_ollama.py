import ssl
import urllib.error
import urllib.request
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from deepagents import create_deep_agent

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text


# 로컬 Ollama 모델 설정
model = ChatOllama(
    model="gemma4:e2b",
    temperature=0,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

if __name__ == "__main__":
    print("Running create_agent...", flush=True)
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}},
    )
    print("Running create_deep_agent...", flush=True)
    deep_agent_result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-da"}},
    )
    print("\ncreate_agent:")
    print(agent_result["messages"][-1].content_blocks)
    print("\ncreate_deep_agent:")
    print(deep_agent_result["messages"][-1].content_blocks)

