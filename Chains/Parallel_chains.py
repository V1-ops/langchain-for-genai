from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel
from dotenv import load_dotenv
load_dotenv()
llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=500
)
llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=500
)
model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
prompt1 = PromptTemplate(
    template = "Generate short and simple notes about the given text :\n{text}",
    input_variables = ["text"]
)
prompt2 = PromptTemplate(
    template = "Generate 5 question answers from the following text :\n{text}",
    input_variables = ["text"]
)
prompt3 = PromptTemplate(
    template = "Merge the following notes and Q&A into a single document :\n Notes:\n{notes}\nQ&A:\n{qa}",
    input_variables = ["notes","qa"]
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser,
    "qa" : prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
text = """At 15 °C ambient temperature, taking a cold bath is generally not recommended, especially if your body is already exposed to cool weather.

Why you should avoid a cold bath at 15 °C

❄️ Risk of excessive body cooling: Your core temperature can drop too fast.

🤧 Higher chances of cold, cough, or fever, particularly if your immunity is low.

🫀 Cold shock response: Sudden cold water can cause rapid breathing and increased heart rate.

💪 Muscle stiffness: Cold tightens muscles, which is bad if you’re exercising or already sore.

When it might be okay

You can consider it only if:

You are healthy, not sick or fatigued

The water is not ice-cold (slightly cool, not very cold)

You warm up immediately after (dry clothes, light movement)

Better alternatives (recommended)

🚿 Lukewarm or warm bath (best option in this weather)

🚿 Normal-temperature bath, then finish with 30 seconds of cooler water if you like the cold-bath feeling

Special note for you

Since you’ve had physiotherapy for a shoulder/rhomboid pinched nerve, cold exposure can increase muscle tightness, so avoid cold baths right now.

Conclusion:
👉 No, avoid a cold bath at 15 °C. Go for lukewarm or warm water.

If you want, tell me why you’re considering a cold bath (recovery, discipline, skin care, fat loss, etc.), and I’ll guide you properly.
Yes 👍 you can go slowly toward cold water, and that’s the safest way to do it—especially in 15 °C weather.

How to do it safely (step-by-step)

🚿 Start with lukewarm water (very important).

After 2–3 minutes, slightly reduce the temperature.

Stay there for 30–60 seconds.

If comfortable, reduce again gradually, not suddenly.

Stop before it feels shocking or uncomfortable.

👉 Total cold exposure should be 1–2 minutes max in this weather.

Important safety rules

❌ Never start directly with cold water

❌ Don’t pour cold water on head first

❌ Stop immediately if you feel shivering, dizziness, or breathlessness

✅ Dry quickly and wear warm clothes after

✅ Do light movement (walking/stretching) to re-warm

For you specifically

Because of your shoulder/rhomboid nerve issue, keep the cold water away from upper back and neck, or keep it very mild there. Cold can increase stiffness.

Best compromise option (recommended)

Normal or warm bath

End with 20–30 seconds of cool (not cold) water on legs/arms only

Bottom line

✔️ Yes, gradual cold exposure is okay
❌ Avoid extreme cold or long duration
✅ Listen to your body
"""


result=chain.invoke ({"text": text})

print(result)
chain.get_graph().print_ascii()


"""+-------------------------+
              | Parallel<notes,qa>Input |
              +-------------------------+
                  ***               ***
               ***                     ***
             **                           **
+----------------+                    +----------------+
| PromptTemplate |                    | PromptTemplate |
+----------------+                    +----------------+
          *                                   *
          *                                   *
          *                                   *
+-----------------+                  +-----------------+
| ChatHuggingFace |                  | ChatHuggingFace |
+-----------------+                  +-----------------+
          *                                   *
          *                                   *
          *                                   *
+-----------------+                  +-----------------+
| StrOutputParser |                  | StrOutputParser |
+-----------------+                  +-----------------+
                  ***               ***
                     ***         ***
                        **     **
              +--------------------------+
              | Parallel<notes,qa>Output |
              +--------------------------+
                            *
                            *
                            *
                   +----------------+
                   | PromptTemplate |
                   +----------------+
                            *
                            *
                            *
                  +-----------------+
                  | ChatHuggingFace |
                  +-----------------+
                            *
                            *
                            *
                  +-----------------+
                  | StrOutputParser |
                  +-----------------+
                            *
                            *
                            *
                +-----------------------+
                | StrOutputParserOutput |
                +-----------------------+
"""
