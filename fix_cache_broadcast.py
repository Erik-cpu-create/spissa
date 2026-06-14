with open("crates/rllm-runtime/src/llama/generate.rs", "r") as f:
    content = f.read()

old_block = """    let cache_broadcasted = if let Some(c) = cache.as_ref() {
        if config.q_heads != config.kv_heads {
            let b_k = broadcast_gqa(c.keys(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
            let b_v = broadcast_gqa(c.values(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
            let mut new_c = KvCache::new(config.q_heads, config.head_dim, c.max_seq_len())?;
            new_c.append(&b_k, &b_v, c.len())?;
            Some(new_c)
        } else {
            None
        }
    } else {
        None
    };"""

new_block = """    let cache_broadcasted = if let Some(c) = cache.as_ref() {
        if config.q_heads != config.kv_heads {
            if c.len() > 0 {
                let b_k = broadcast_gqa(c.keys(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
                let b_v = broadcast_gqa(c.values(), c.len(), config.kv_heads, config.q_heads, config.head_dim)?;
                let mut new_c = KvCache::new(config.q_heads, config.head_dim, c.max_seq_len())?;
                new_c.append(&b_k, &b_v, c.len())?;
                Some(new_c)
            } else {
                Some(KvCache::new(config.q_heads, config.head_dim, c.max_seq_len())?)
            }
        } else {
            None
        }
    } else {
        None
    };"""

content = content.replace(old_block, new_block)

with open("crates/rllm-runtime/src/llama/generate.rs", "w") as f:
    f.write(content)
