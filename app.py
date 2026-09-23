import os
import subprocess
import tempfile
import traceback
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. Define Verilog Tools
# ============================================================

@tool
def generate_verilog_testbench(verilog_code: str) -> str:
    """Generate a Verilog testbench for the given Verilog design."""
    prompt = f"""
You are a Verilog verification engineer.

Generate a complete Verilog testbench for this design:

{verilog_code}

Requirements:
1. Use Verilog only, not Python.
2. Do not use SystemVerilog.
3. Identify the DUT module and its ports correctly.
4. Instantiate the DUT.
5. Apply normal test cases and edge cases.
6. Use $display to show important outputs.
7. Use $finish to stop simulation.
8. Return ONLY the testbench code.
9. Do not use markdown code fences.
"""
    response = llm.invoke(prompt)
    return clean_code(response.content)


@tool
def run_verilog(verilog_code: str, testbench_code: str) -> str:
    """Compile and simulate Verilog using Icarus Verilog."""
    try:
        verilog_code = clean_code(verilog_code)
        testbench_code = clean_code(testbench_code)

        with tempfile.TemporaryDirectory() as temp_dir:
            design_file = os.path.join(temp_dir, "design.v")
            tb_file = os.path.join(temp_dir, "testbench.v")
            output_file = os.path.join(temp_dir, "simulation.out")

            with open(design_file, "w", encoding="utf-8") as f:
                f.write(verilog_code)

            with open(tb_file, "w", encoding="utf-8") as f:
                f.write(testbench_code)

            compile_result = subprocess.run(
                [
                    "iverilog",
                    "-o",
                    output_file,
                    design_file,
                    tb_file
                ],
                capture_output=True,
                text=True
            )

            if compile_result.returncode != 0:
                return (
                    "VERILOG COMPILATION FAILED\n\n"
                    + compile_result.stderr
                )

            simulation_result = subprocess.run(
                ["vvp", output_file],
                capture_output=True,
                text=True
            )

            if simulation_result.returncode != 0:
                return (
                    "VERILOG SIMULATION FAILED\n\n"
                    + simulation_result.stderr
                )

            output = simulation_result.stdout.strip()

            if not output:
                output = "Simulation completed successfully with no console output."

            return (
                "VERILOG SIMULATION SUCCESSFUL\n\n"
                "SIMULATION OUTPUT:\n"
                + output
            )

    except FileNotFoundError:
        return (
            "Icarus Verilog is not installed or is not available in PATH. "
            "Install Icarus Verilog and make sure 'iverilog' and 'vvp' "
            "commands are available."
        )

    except Exception:
        return "Execution error:\n" + traceback.format_exc()


@tool
def verify_verilog(verilog_code: str) -> str:
    """Generate a testbench and execute the Verilog design."""
    testbench = generate_verilog_testbench.invoke(verilog_code)
    result = run_verilog.invoke({
        "verilog_code": verilog_code,
        "testbench_code": testbench
    })

    return (
        "GENERATED TESTBENCH:\n"
        + testbench
        + "\n\n"
        + result
    )


tools = [
    generate_verilog_testbench,
    run_verilog,
    verify_verilog
]


# ============================================================
# 2. Helper Function
# ============================================================

def clean_code(content) -> str:
    """Remove markdown code fences from model output."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    content = str(content)

    content = content.replace("```verilog", "")
    content = content.replace("```Verilog", "")
    content = content.replace("```", "")

    return content.strip()


# ============================================================
# 3. Initialize Gemini Model & Agent
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a specialized Verilog RTL and verification agent. "
        "You are restricted ONLY to Verilog, RTL design, digital logic, "
        "Verilog testbenches, simulation, and hardware-description-language "
        "coding tasks. "
        "For unrelated topics, say exactly: "
        "'I am not authorized to answer questions outside of Verilog and digital design.' "
        "When the user asks for Verilog code, provide complete Verilog code. "
        "When useful, use the verification tools to generate a testbench "
        "and simulate the design using Icarus Verilog."
    )
)


# ============================================================
# 4. FastAPI Input Model
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(description="Your Verilog task")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)

        if content is not None:
            if isinstance(content, list):
                parts = []

                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))

                return "\n".join(parts)

            return str(content)

        return str(last)

    return str(agent_output)


# ============================================================
# 5. LangServe Chain
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 6. FastAPI App
# ============================================================

app = FastAPI(
    title="Verilog AI Agent",
    description="Gemini-powered Verilog RTL generation and simulation agent",
    version="1.0.0"
)

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


@app.get("/")
def home():
    return {
        "message": "Verilog AI Agent is running",
        "endpoint": "/agent"
    }


# ============================================================
# 7. Run Server
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
