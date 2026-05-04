from agent import ask_agent
from tools import run_code

# 🔥 Strict prompt for EXACT 3 graphs
prompt = """
Write Python code using pandas and matplotlib.

Load dataset from 'data/sample.csv'.

You MUST create EXACTLY 3 different charts:

1. Line chart of Sales vs Month
   - save as outputs/line_chart.png

2. Bar chart of Sales vs Month
   - save as outputs/bar_chart.png

3. Histogram of Sales
   - save as outputs/histogram.png

IMPORTANT RULES:
- Do NOT use plt.show()
- Before EACH chart, use plt.figure()
- After EACH chart, use plt.savefig() with correct filename
- Do NOT overwrite files
- Ensure all 3 charts are saved separately
- Only return Python code (no explanation)
"""

# get code from AI
generated_code = ask_agent(prompt)

print("Generated Code:\n", generated_code)

# run generated code
result = run_code(generated_code)

print("Execution Result:", result)