with open("crates/rllm-runtime/src/llama/generate.rs", "r") as f:
    content = f.read()

content = content.replace("    } else {\n        k\n    };", "    } else {\n        k.clone()\n    };")
content = content.replace("    } else {\n        v\n    };", "    } else {\n        v.clone()\n    };")

with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
    f.write(content)
