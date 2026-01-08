


┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: DEFINE TEMPLATE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  template = PromptTemplate(                                 │
│      template=...{paper_input}...{style_input}...,   │
│      input_variables=["paper_input", "style_input", ...]   │
│  )                                                          │
│                                                              │
│  ✅ input_variables = List of placeholder names             │
│     └─ Tells LangChain: "I will use these 3 placeholders"  │
│     └─ Must MATCH the {placeholders} in template string    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: GET VALUES FROM STREAMLIT              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  paper_input = "Attention Is All You Need"  (from selectbox)│
│  style_input = "Beginner-Friendly"          (from selectbox)│
│  length_input = "Short (1-2 paragraphs)"    (from selectbox)│
│                                                              │
│  ✅ These are actual VALUES selected by user                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          STEP 3: RENAME FOR CLARITY (OPTIONAL)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  selected_paper = paper_input                               │
│  selected_style = style_input                               │
│  selected_length = length_input                             │
│                                                              │
│  ✅ Same values, clearer names                              │
│  ✅ Shows intent: "these are user's selections"             │
│  ✅ Makes code more readable                                │
│                                                              │
│  Note: This step is OPTIONAL - you could skip it!           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│        STEP 4: MAP VALUES TO PLACEHOLDERS (FORMAT)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  prompt = template.format(                                  │
│      paper_input=selected_paper,        ← Placeholder name  │
│      style_input=selected_style,        ← Placeholder name  │
│      length_input=selected_length       ← Placeholder name  │
│  )                                                          │
│                                                              │
│  Mapping Process:                                           │
│  ┌──────────────────┬──────────────────┬─────────────────┐ │
│  │ Placeholder Name │ Variable Value   │ Result          │ │
│  ├──────────────────┼──────────────────┼─────────────────┤ │
│  │ paper_input      │ selected_paper   │ "Attention...  │ │
│  │ style_input      │ selected_style   │ "Beginner-     │ │
│  │ length_input     │ selected_length  │ "Short (1-2)   │ │
│  └──────────────────┴──────────────────┴─────────────────┘ │
│                                                              │
│  ✅ template.format() replaces all {placeholders}          │
│  ✅ Returns a complete prompt string ready for model        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 5: SEND TO MODEL & GET RESPONSE           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  result = model.invoke(prompt)                              │
│                                                              │
│  ✅ model receives the complete formatted prompt            │
│  ✅ Returns: AIMessage with model's response                │
│  ✅ Display with: st.write(result.content)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
"""
