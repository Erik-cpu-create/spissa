with open("crates/rllm-runtime/src/rotary.rs", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "rotate_llama_pair(q, row_start, pair, cos, sin);" in line:
        new_lines.append(line.replace("rotate_llama_pair(q, row_start, pair, cos, sin);", "rotate_neox_pair(q, row_start, pair, half_rotary, cos, sin);"))
    elif "rotate_llama_pair(k, row_start, pair, cos, sin);" in line:
        new_lines.append(line.replace("rotate_llama_pair(k, row_start, pair, cos, sin);", "rotate_neox_pair(k, row_start, pair, half_rotary, cos, sin);"))
    else:
        new_lines.append(line)

with open("crates/rllm-runtime/src/rotary.rs", "w") as f:
    f.writelines(new_lines)

