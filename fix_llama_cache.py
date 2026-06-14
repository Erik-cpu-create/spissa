with open("crates/rllm-runtime/src/llama/generate.rs", "r") as f:
    content = f.read()

# We need to move c.append(&k, &v, config.seq_len)? to AFTER attention is computed.

# Find the block where cache is appended
append_str = """    if let Some(c) = cache.as_deref_mut() {
        c.append(&k, &v, config.seq_len)?;
    }"""

# Remove it
content = content.replace(append_str, "")

# Add it after attn_out
attn_out_str = """    let attn_out = if let Some(c) = cache_ref {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, Some(c), attn_config)?
    } else if config.q_heads == config.kv_heads && cache.is_some() {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, cache.as_deref(), attn_config)?
    } else {
        scaled_dot_product_attention_with_cache(&q, &k_broadcasted, &v_broadcasted, None, attn_config)?
    };"""

new_attn_out = attn_out_str + """\n
    if let Some(c) = cache.as_deref_mut() {
        c.append(&k, &v, config.seq_len)?;
    }"""

content = content.replace(attn_out_str, new_attn_out)

with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
    f.write(content)
