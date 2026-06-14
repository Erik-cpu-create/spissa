import sys

def main():
    # 1. model.rs
    with open("crates/rllm-runtime/src/llama/model.rs", "r") as f:
        model_rs = f.read()
    model_rs = model_rs.replace("use crate::{Result, StreamingSamplingConfig};", "use crate::StreamingSamplingConfig;")
    with open("crates/rllm-runtime/src/llama/model.rs", "w") as f:
        f.write(model_rs)

    # 2. generate.rs
    with open("crates/rllm-runtime/src/llama/generate.rs", "r") as f:
        gen_rs = f.read()
    gen_rs = gen_rs.replace("use std::time::Instant;\n", "")
    gen_rs = gen_rs.replace("let mut v =", "let v =")
    gen_rs = gen_rs.replace("cache: Option<&mut KvCache>", "mut cache: Option<&mut KvCache>")
    with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
        f.write(gen_rs)

    # 3. api.rs
    with open("crates/rllm-runtime/src/llama/api.rs", "r") as f:
        api_rs = f.read()
    api_rs = api_rs.replace("use rllm_container::GlobalMetadata;\n", "")
    api_rs = api_rs.replace("use std::time::Instant;\n", "")
    api_rs = api_rs.replace("let intermediate_size = model_config.intermediate_size.unwrap() as usize;\n", "")
    api_rs = api_rs.replace("let head_dim = hidden_size / num_heads;\n", "")
    api_rs = api_rs.replace("options: LlamaRamaGenerationOptions,", "_options: LlamaRamaGenerationOptions,")
    with open("crates/rllm-runtime/src/llama/api.rs", "w") as f:
        f.write(api_rs)

if __name__ == "__main__":
    main()
