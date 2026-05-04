import matplotlib.pyplot as plt
import os

def run_code(code):
    try:
        os.makedirs("outputs", exist_ok=True)

        # remove markdown
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code.replace("python", "", 1)

        code = code.strip()

        # 🔥 FORCE FIXES (VERY IMPORTANT)

        # remove plt.show()
        code = code.replace("plt.show()", "")

        # ❌ remove hardcoded query
        code = code.replace('query = "show trend"', "")
        code = code.replace('query="show trend"', "")

        # ❌ remove bad plotting loops
        code = code.replace("for col in numeric_cols:", "for col in numeric_cols[:2]:")

        # execute
        exec_globals = {}
        exec(code, exec_globals)

        return "Code executed successfully"

    except Exception as e:
        return str(e)