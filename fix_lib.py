import re

with open("crates/rllm-runtime/src/lib.rs", "r") as f:
    content = f.read()

content = content.replace("mod gpt_neox;\n", "pub mod models;\n")
content = content.replace("pub use llama::*;\n", "pub use models::llama::*;\n")
content = re.sub(r"pub use gpt_neox::\{[^}]+\};\n", "pub use models::gpt_neox::*;\n", content, flags=re.MULTILINE | re.DOTALL)

with open("crates/rllm-runtime/src/lib.rs", "w") as f:
    f.write(content)
