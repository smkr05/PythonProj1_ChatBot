from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2:1b")

template = """

You are an expert in answering questions about cars based on customer reviews.
Here are some relevant reviews to base answers off {reviews}
Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------")
    question = input("Please enter your question about the pizza place: (type q to quit)\n")
    print("\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
