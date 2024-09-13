from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

### Buat LLM dengan LLama 3.1 dari LM studio
llm = ChatOpenAI(base_url="http://192.168.88.83:1234/v1/", api_key="random_api_key", model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", streaming=True)

# Membuat prompt untuk debat
prompt_debat = ChatPromptTemplate.from_template(
    """<s>Sebuah debat antara dua orang yang memiliki pendapat berbeda tentang topik {topic}</s>
    <s>Pihak 1 (Asisten 1): {pilihan_pihak_1}</s>
    <s>Pihak 2 (Asisten 2): {pilihan_pihak_2}</s>
    """
)

# Membuat prompt untuk menanyakan pertanyaan
prompt_tanya = ChatPromptTemplate.from_template(
    """<s>Sebuah percakapan antara pengguna yang penasaran dan asisten kecerdasan buatan. Asisten memberikan jawaban yang membantu, rinci, dan sopan atas pertanyaan pengguna.</s>
    <s>Manusia: {question}</s>
    Asisten:
    """
)

# Membuat chain untuk debat
chain_debat = prompt_debat | llm | StrOutputParser()

# Membuat chain untuk menanyakan pertanyaan
chain_tanya = prompt_tanya | llm | StrOutputParser()

def stream_debat(topic, pilihan_pihak_1, pilihan_pihak_2):
    for s in chain_debat.invoke({"topic": topic, "pilihan_pihak_1": pilihan_pihak_1, "pilihan_pihak_2": pilihan_pihak_2}):
        print(s, end="", flush=True)

def stream_tanya(question):
    for s in chain_tanya.invoke({"question": question}):
        print(s, end="", flush=True)

stream_debat("Apakah Indonesia harus meningkatkan investasi pada teknologi?", "Ya, karena teknologi dapat mempercepat kemajuan negara", "Tidak, karena investasi pada teknologi dapat meningkatkan kesenjangan sosial")