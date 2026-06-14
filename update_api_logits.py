import sys

with open("crates/rllm-runtime/src/llama/api.rs", "r") as f:
    api_rs = f.read()

# Add logits to LlamaTextGenerationResult
api_rs = api_rs.replace(
    "pub struct LlamaTextGenerationResult {",
    "pub struct LlamaTextGenerationResult {\n    pub logits: Option<Vec<f32>>,"
)

# In api.rs, the result creation:
old_result = """    Ok(LlamaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        text: String::new(),
        generated_text: String::new(),
        context_echo_bytes: 0,
    })"""
new_result = """    Ok(LlamaTextGenerationResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        generated_token_ids,
        token_ids,
        text: String::new(),
        generated_text: String::new(),
        context_echo_bytes: 0,
        logits: if _options.collect_logits { Some(logits.clone()) } else { None },
    })"""
api_rs = api_rs.replace(old_result, new_result)

with open("crates/rllm-runtime/src/llama/api.rs", "w") as f:
    f.write(api_rs)

with open("crates/rllm-runtime/src/llama/model.rs", "r") as f:
    model_rs = f.read()

model_rs = model_rs.replace(
    """pub struct LlamaTextGenerationResult {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_ids: Vec<usize>,
    pub text: String,
    pub generated_text: String,
    pub context_echo_bytes: usize,
}""",
    """pub struct LlamaTextGenerationResult {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub token_ids: Vec<usize>,
    pub text: String,
    pub generated_text: String,
    pub context_echo_bytes: usize,
    pub logits: Option<Vec<f32>>,
}"""
)

with open("crates/rllm-runtime/src/llama/model.rs", "w") as f:
    f.write(model_rs)
