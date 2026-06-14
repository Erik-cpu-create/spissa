import sys

def main():
    # 1. Fix generate.rs
    with open("crates/rllm-runtime/src/llama/generate.rs", "r") as f:
        generate_code = f.read()
    
    generate_code = generate_code.replace("use crate::{", "use crate::{StreamingLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,")
    
    # Fix the linear config
    for old, new in [
        (
            "let q_config = StreamingTileLinearConfig {",
            "let q_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        ),
        (
            "let kv_config = StreamingTileLinearConfig {",
            "let kv_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        ),
        (
            "let o_config = StreamingTileLinearConfig {",
            "let o_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        ),
        (
            "let mlp_config = StreamingTileLinearConfig {",
            "let mlp_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        ),
        (
            "let down_config = StreamingTileLinearConfig {",
            "let down_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        ),
        (
            "let lm_config = StreamingTileLinearConfig {",
            "let lm_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {"
        )
    ]:
        generate_code = generate_code.replace(old, new)
        
    generate_code = generate_code.replace(
        "    };",
        "    }, tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS };"
    )

    # Fix streaming_tile_linear_from_model signature:
    # model, weight_name, input, bias, config, budget
    generate_code = generate_code.replace(
        "streaming_tile_linear_from_model(model, &attention_input, &names.q_weight, q_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.q_weight, &attention_input, None, q_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &attention_input, &names.k_weight, kv_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.k_weight, &attention_input, None, kv_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &attention_input, &names.v_weight, kv_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.v_weight, &attention_input, None, kv_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &attn_out, &names.o_weight, o_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.o_weight, &attn_out, None, o_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &mlp_input, &names.gate_weight, mlp_config.clone(), budget)?",
        "streaming_tile_linear_from_model(model, &names.gate_weight, &mlp_input, None, mlp_config.clone(), budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &mlp_input, &names.gate_weight, mlp_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.gate_weight, &mlp_input, None, mlp_config.clone(), budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &mlp_input, &names.up_weight, mlp_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.up_weight, &mlp_input, None, mlp_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, &gate, &names.down_weight, down_config, budget)?",
        "streaming_tile_linear_from_model(model, &names.down_weight, &gate, None, down_config, budget)?"
    ).replace(
        "streaming_tile_linear_from_model(model, last_hidden, &prepared.lm_head_weight, lm_config, budget)?",
        "streaming_tile_linear_from_model(model, &prepared.lm_head_weight, last_hidden, None, lm_config, budget)?"
    )

    with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
        f.write(generate_code)

    # 2. Fix api.rs
    with open("crates/rllm-runtime/src/llama/api.rs", "r") as f:
        api_code = f.read()
    
    api_code = api_code.replace("use crate::{", "use crate::{StreamingLinearConfig, DEFAULT_STREAMING_TILE_ELEMENTS,")
    
    # define decode_vector_tensor helper inside api.rs or use it from gpt_neox? 
    # it's not pub. We will define it inside api.rs
    
    api_code = api_code.replace("use rllm_container::GlobalMetadata;", """use rllm_container::GlobalMetadata;

fn decode_vector_tensor(
    model: &mut LazyRllmModel,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let mut budget = MemoryBudget::unbounded();
    let tensor = model.decode_tensor(name, &mut budget)?;
    if tensor.shape != [expected_len] {
        return Err(RuntimeError::Shape(format!(
            "tensor {name} shape {:?} does not match expected [{expected_len}]",
            tensor.shape
        )));
    }
    Ok(tensor.data)
}
""")

    api_code = api_code.replace("model.load_layer_tensor(\"model.norm.weight\")?.to_f32_vec()?", "decode_vector_tensor(model, \"model.norm.weight\", hidden_size)?")
    api_code = api_code.replace("model.has_tensor", "model.tensor(name).is_ok()")
    api_code = api_code.replace(
        """while model.tensor(name).is_ok()(&format!("model.layers.{num_layers}.input_layernorm.weight")) {""",
        """while model.tensor(&format!("model.layers.{num_layers}.input_layernorm.weight")).is_ok() {"""
    )
    
    api_code = api_code.replace(
        "model.load_layer_tensor(&prepared.embedding_weight)?.to_f32_vec()?",
        "model.decode_tensor(&prepared.embedding_weight, budget)?.data"
    )
    api_code = api_code.replace(
        "model.load_layer_tensor(&format!(\"model.layers.{i}.input_layernorm.weight\"))?.to_f32_vec()?",
        "decode_vector_tensor(model, &format!(\"model.layers.{i}.input_layernorm.weight\"), hidden_size)?"
    )
    api_code = api_code.replace(
        "model.load_layer_tensor(&format!(\"model.layers.{i}.post_attention_layernorm.weight\"))?.to_f32_vec()?",
        "decode_vector_tensor(model, &format!(\"model.layers.{i}.post_attention_layernorm.weight\"), hidden_size)?"
    )

    api_code = api_code.replace("let lm_config = StreamingTileLinearConfig {", "let lm_config = StreamingTileLinearConfig { linear: StreamingLinearConfig {")
    api_code = api_code.replace(
        """            in_features: hidden_size,
            out_features: vocab_size,
        };""",
        """            in_features: hidden_size,
            out_features: vocab_size,
        }, tile_elements: DEFAULT_STREAMING_TILE_ELEMENTS };"""
    )
    api_code = api_code.replace(
        "streaming_tile_linear_from_model(model, last_hidden, &prepared.lm_head_weight, lm_config, budget)?",
        "streaming_tile_linear_from_model(model, &prepared.lm_head_weight, last_hidden, None, lm_config, budget)?"
    )
    
    # Wait, the budget needs to be passed to model.decode_tensor
    # Let's write the whole api.rs again to be safe.
    
    with open("crates/rllm-runtime/src/llama/api.rs", "w") as f:
        f.write(api_code)

if __name__ == "__main__":
    main()
