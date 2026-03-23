import os
import re
import csv
import numpy as np
import torch


# -----------------------------
# Fixed-point helpers
# -----------------------------
def float_to_fixed_int32(x: np.ndarray, frac_bits: int) -> np.ndarray:
    """Round-to-nearest and saturate to int32."""
    x = np.asarray(x, dtype=np.float64)
    scaled = np.round(x * (1 << frac_bits)).astype(np.int64)
    scaled = np.clip(scaled, -2**31, 2**31 - 1)
    return scaled.astype(np.int32)


def lanes_to_hex_word(lanes_i32: np.ndarray, little_endian=True) -> str:
    """
    Pack P lanes of int32 into one wide hex word.
    little_endian=True => lane0 is LSB (matches w_lane[p] = vec[p*32 +:32])
    """
    lanes_i32 = np.asarray(lanes_i32, dtype=np.int32)
    P = lanes_i32.size
    word = 0
    if little_endian:
        for p in range(P):
            word |= (int(np.uint32(lanes_i32[p])) & 0xFFFFFFFF) << (32 * p)
    else:
        for p in range(P):
            word |= (int(np.uint32(lanes_i32[p])) & 0xFFFFFFFF) << (32 * (P - 1 - p))
    width_hex = (32 * P) // 4
    return f"{word:0{width_hex}X}"


def u64_to_hex_word(x_u64: np.uint64) -> str:
    return f"{int(x_u64):016X}"


# -----------------------------
# Packing helpers
# -----------------------------
def pack_conv_parallel_filters(weights_oc_ke: np.ndarray, P_FILTER: int) -> np.ndarray:
    """
    weights_oc_ke: shape (OUT_C, WIN_ELEMS) int32
    returns packed: shape (DEPTH, P_FILTER) int32 where
      DEPTH = (OUT_C/P_FILTER) * WIN_ELEMS
      addr = f_group*WIN_ELEMS + k
      packed[addr, lane] = weights[f_group*P_FILTER + lane, k]
    """
    weights_oc_ke = np.asarray(weights_oc_ke)
    if weights_oc_ke.ndim != 2:
        raise ValueError(f"Expected (OUT_C, WIN_ELEMS), got {weights_oc_ke.shape}")

    OUT_C, WIN_ELEMS = weights_oc_ke.shape
    if OUT_C % P_FILTER != 0:
        raise ValueError(f"OUT_C={OUT_C} must be divisible by P_FILTER={P_FILTER}")

    F_GROUPS = OUT_C // P_FILTER
    DEPTH = F_GROUPS * WIN_ELEMS
    packed = np.zeros((DEPTH, P_FILTER), dtype=np.int32)

    for fg in range(F_GROUPS):
        base_f = fg * P_FILTER
        base_addr = fg * WIN_ELEMS
        # vectorize over k
        for k in range(WIN_ELEMS):
            addr = base_addr + k
            packed[addr, :] = weights_oc_ke[base_f:base_f + P_FILTER, k]

    return packed

def pack_fc_weights_to_coe_lines(
    weights,          # shape: (OUT_C, IN_C)
    parallel_filters=5
):
    """
    Packs FC weights into 256-bit lines for FPGA BRAM.

    Output format matches RTL:
    address = group * IN_C + input_channel

    Each line = 8 × 32-bit weights (one per output lane)
    """

    OUT_C, IN_C = weights.shape
    P = parallel_filters

    assert OUT_C % P == 0, "OUT_C must be divisible by P_FILTER"

    groups = OUT_C // P
    lines = []

    for g in range(groups):
        for ic in range(IN_C):

            lanes = np.zeros(P, dtype=np.uint32)

            for p in range(P):
                oc = g * P + p
                val = weights[oc, ic]

                lanes[p] = np.uint32(val)

            # pack into hex string (lane0 = LSB)
            word = ''.join(f'{x:08X}' for x in lanes[::-1])
            lines.append(word)

    return lines

def pack_bn_ab_bmg(a_i32: np.ndarray,
                   b_i32: np.ndarray,
                   p_filter: int = 8,
                   little_endian: bool = True):
    """
    Pack BN parameters for BMG.

    Per channel:
        A = int32
        B = int32
        packed = 64 bits

    BMG word:
        p_filter channels packed together.

    Example (p_filter=8):
        word[63:0]    -> channel0
        word[127:64]  -> channel1
        ...
        word[511:448] -> channel7

    Returns
    -------
    np.ndarray(dtype=object)
        Array of Python integers representing 512-bit words
    """
    a_i32 = np.asarray(a_i32, dtype=np.int32).reshape(-1)
    b_i32 = np.asarray(b_i32, dtype=np.int32).reshape(-1)

    if a_i32.size != b_i32.size:
        raise ValueError("a and b must have same length")

    out_c = a_i32.size

    if out_c % p_filter != 0:
        raise ValueError("OUT_C must be divisible by p_filter")

    groups = out_c // p_filter

    words = []

    for g in range(groups):

        word = 0

        for lane in range(p_filter):

            idx = g * p_filter + lane

            au = int(a_i32[idx]) & 0xFFFFFFFF
            bu = int(b_i32[idx]) & 0xFFFFFFFF

            if little_endian:
                pair = (au << 32) | bu
            else:
                pair = (bu << 32) | au

            word |= pair << (lane * 64)

        words.append(word)

    return np.array(words, dtype=object)

def pack_bias_parallel_filters(bias_i32: np.ndarray, P_FILTER: int) -> np.ndarray:
    """
    bias_i32: shape (OUT_C,) int32
    Returns packed: shape (F_GROUPS, P_FILTER) int32

    addr = f_group
    packed[addr, lane] = bias[f_group*P_FILTER + lane]
    """
    bias_i32 = np.asarray(bias_i32, dtype=np.int32).reshape(-1)

    OUT_C = bias_i32.size
    if OUT_C % P_FILTER != 0:
        raise ValueError(f"OUT_C={OUT_C} must be divisible by P_FILTER={P_FILTER}")

    F_GROUPS = OUT_C // P_FILTER
    packed = np.zeros((F_GROUPS, P_FILTER), dtype=np.int32)

    for fg in range(F_GROUPS):
        base = fg * P_FILTER
        packed[fg, :] = bias_i32[base:base + P_FILTER]

    return packed

# -----------------------------
# File writers
# -----------------------------
def write_bin_int32(path, data_i32: np.ndarray):
    data_i32 = np.asarray(data_i32, dtype=np.int32).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_i32.tofile(path)


def write_bin_u64(path, data_u64: np.ndarray):
    data_u64 = np.asarray(data_u64, dtype=np.uint64).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_u64.tofile(path)


def write_coe_wide_from_packed_i32(path, packed_depth_by_lanes: np.ndarray, little_endian=True):
    packed = np.asarray(packed_depth_by_lanes, dtype=np.int32)
    DEPTH, P = packed.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for addr in range(DEPTH):
            hx = lanes_to_hex_word(packed[addr, :], little_endian=little_endian)
            f.write(hx + (",\n" if addr != DEPTH - 1 else ";\n"))


def write_coe_u64(path, words_u64: np.ndarray):
    words = np.asarray(words_u64, dtype=np.uint64).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i in range(words.size):
            hx = u64_to_hex_word(words[i])
            f.write(hx + (",\n" if i != words.size - 1 else ";\n"))


def write_csv_matrix(path, mat: np.ndarray, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        if header:
            w.writerow([header])
        mat = np.asarray(mat)
        if mat.ndim == 1:
            for v in mat:
                w.writerow([int(v)])
        elif mat.ndim == 2:
            for r in range(mat.shape[0]):
                w.writerow([int(x) for x in mat[r, :]])
        else:
            w.writerow([f"Unsupported shape {mat.shape}"])


def write_csv_bn(path, a_f, b_f, a_i32, b_i32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        # w.writerow(["ch", "a_float", "b_float", "a_i32(Q2.30)", "b_i32(Q12.20)"])
        w.writerow(["ch", "a_float", "b_float", "a_i32(Q12.20)", "b_i32(Q12.20)"])
        for i in range(len(a_f)):
            w.writerow([i, float(a_f[i]), float(b_f[i]), int(a_i32[i]), int(b_i32[i])])


def bitpack_binary_weights_to_u32(words01: np.ndarray) -> np.ndarray:
    """
    words01: 0/1 array
    returns u32 array where bit0 corresponds to first element of words01 (LSB-first within u32).
    """
    bits = np.asarray(words01, dtype=np.uint8).reshape(-1)
    nbits = bits.size
    nwords = (nbits + 31) // 32
    out = np.zeros(nwords, dtype=np.uint32)
    for i in range(nbits):
        if bits[i]:
            out[i // 32] |= (1 << (i % 32))
    return out

def pack_binary_weights_to_coe_lines(
        flat_bits,
        Cin=64,
        Cout=64,
        K=3,
        parallel_filters=8
):
    """
    Convert flattened binary weights (0/1) into 256-bit COE lines
    compatible with your RTL weight BMG.

    flat_bits shape: (K*K*Cin*Cout,) or (36864,1)

    Returns: list of hex strings
    """

    bits = np.asarray(flat_bits, dtype=np.int8).reshape(-1)

    # restore convolution layout
    weights = bits.reshape(Cout, Cin, K, K)

    cin_groups = Cin // 32
    P = parallel_filters

    lines = []

    for oc_group in range(Cout // P):

        for ky in range(K):
            for kx in range(K):
                for cg in range(cin_groups):

                    lanes = np.zeros(P, dtype=np.uint32)

                    for p in range(P):

                        oc = oc_group * P + p

                        packed = 0

                        for b in range(32):

                            c = cg * 32 + b

                            if weights[oc, c, ky, kx] > 0:
                                packed |= (1 << (31 - b))

                        lanes[p] = np.uint32(packed)

                    lines.append(lanes_to_hex_word(lanes))

    return lines

def write_coe_u32(path, data_u32: np.ndarray):
    data = np.asarray(data_u32, dtype=np.uint32).reshape(-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i in range(data.size):
            hx = f"{int(data[i]):08X}"
            f.write(hx + (",\n" if i != data.size - 1 else ";\n"))

def write_coe_from_lines(filename, lines):

    with open(filename, "w") as f:

        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")

        for i, line in enumerate(lines):

            if i != len(lines) - 1:
                f.write(line + ",\n")
            else:
                f.write(line + ";\n")

def write_coe(filename, words, width=512):

    hex_width = width // 4

    with open(filename, "w") as f:

        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")

        for i, w in enumerate(words):

            hex_word = format(w, f"0{hex_width}x")

            if i != len(words) - 1:
                f.write(hex_word + ",\n")
            else:
                f.write(hex_word + ";\n")

# -----------------------------
# Name helpers
# -----------------------------
def sanitize_tag(s: str, maxlen=8) -> str:
    # keep stable and filesystem friendly
    s = s.replace("module.", "")
    s = s.replace("layer", "")
    s = s.replace('weight', 'w').replace('bias', 'b')
    s = s.replace('weight', 'w').replace('bias', 'b')
    s = s.replace('running_mean', 'rm').replace('running_var', 'rv')
    s = s.replace('.', '')
    s = s.replace('bn1', 'bn_') # Weight -> Bias -> RM -> RV
    # s = s.replace('_rv', '_WB')
    s = s.upper()

    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


# -----------------------------
# Main exporter
# -----------------------------
def saveWeightsPerLayer_fixedpoint(
    net,
    out_dir="pytorch_implementation/BiReal18_34/savedWeights/WEIGHTS",
    P_FILTER=8,
    conv_frac_bits=20,      # Q12.20
    bn_a_frac_bits=20,      # Q12.20
    bn_b_frac_bits=20,      # Q12.20
    bn_eps=1e-5,
    little_endian=True
):
    """
    Exports per-layer files:
      - Conv weights (packed for parallel filters): .BIN (int32 stream), .COE (wide), .CSV
      - BN fused params: .BIN (uint64), .COE (64-bit), .CSV
      - Binary weights (if present): .BIN (bytes), .COE (u32), .CSV (0/1)
    """
    net = net.cpu()
    sd = net.state_dict()

    # directories
    bin_dir = os.path.join(out_dir, "BIN")
    coe_dir = os.path.join(out_dir, "COE")
    csv_dir = os.path.join(out_dir, "CSV")
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(coe_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # BN staging: collect gamma/beta/rm/rv per BN prefix
    bn_stash = {}  # key -> dict(gamma,beta,rm,rv)

    def bn_key_from_name(n: str) -> str:
        # normalize bn names like "...bn1.weight", "...bn1.bias", "...bn1.running_mean", "...bn1.running_var"
        # key is path without the field
        n = n.replace("module.", "")
        for suf in [".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"]:
            if n.endswith(suf):
                n = n[: -len(suf)]
        return n

    exported = 0
    bnCounter = 0

    prev_fullConvWeights = None
    prev_binaryConvWeights = None

    for name, tensor in sd.items():
        if "num_batches_tracked" in name:
            continue

        tag = sanitize_tag(name)

        # -----------------------------
        # Binary weights (bitpacked)
        # -----------------------------
        if "binary" in name:
            # sign -> {0,1}
            if "layer1.0" in name:
                IN_C = OUT_C
                OUT_C = OUT_C
            elif ".0" in name:
                IN_C = OUT_C
                OUT_C = OUT_C * 2
            else:
                IN_C = OUT_C
                OUT_C = OUT_C

            KW = 3
            KH = 3
            w = torch.sign(tensor).reshape(-1)
            packed_w = pack_binary_weights_to_coe_lines(w, IN_C, OUT_C, 3, 8)
                # bits01 = (w > 0).numpy().astype(np.uint8)

                # # BIN: packed bytes (np.packbits is MSB-first within byte; fine as long as you match in HW)
                # packed_bytes = np.packbits(bits01)
            # bin_path = os.path.join(bin_dir, f"{tag}.BIN")
            # with open(bin_path, "wb") as f:
            #     f.write(packed_w.tobytes())

            # COE: pack into u32 (LSB-first within u32) for easy BRAM init if needed
                # u32_words = bitpack_binary_weights_to_u32(bits01)
                
            coe_path = os.path.join(coe_dir, f"{tag}.COE")
            write_coe_from_lines(coe_path, packed_w)

            # CSV: raw 0/1 list (can be big; still useful for debug)
            # csv_path = os.path.join(csv_dir, f"{tag}.CSV")
            # write_csv_matrix(csv_path, packed_w, header=f"{name} binary bits (0/1)")

            exported += 1
            prev_binaryConvWeights = packed_w
            continue

        # -----------------------------
        # BatchNorm params collection
        # -----------------------------
        if ".downsample.2" in name or "bn" in name:
            # heuristic: capture bn fields
            if name.endswith(".weight"):
                k = bn_key_from_name(name)
                bn_stash.setdefault(k, {})["gamma"] = tensor.detach().numpy()
                continue
            if name.endswith(".bias"):
                k = bn_key_from_name(name)
                bn_stash.setdefault(k, {})["beta"] = tensor.detach().numpy()
                continue
            if name.endswith(".running_mean"):
                k = bn_key_from_name(name)
                bn_stash.setdefault(k, {})["rm"] = tensor.detach().numpy()
                continue
            if name.endswith(".running_var"):
                k = bn_key_from_name(name)
                bn_stash.setdefault(k, {})["rv"] = tensor.detach().numpy()

                # if we now have all 4, export fused BN for this BN block
                bn = bn_stash[k]
                if all(x in bn for x in ["gamma", "beta", "rm", "rv"]):
                    gamma = bn["gamma"]
                    beta  = bn["beta"]
                    rm    = bn["rm"]
                    rv    = bn["rv"]

                    inv = 1.0 / np.sqrt(rv + bn_eps)
                    a_f = gamma * inv
                    b_f = beta - a_f * rm

                    if prev_fullConvWeights is not None:
                        fusedWeights = prev_fullConvWeights * a_f[:, None, None, None]
                        temp = a_f
                        a_f = fusedWeights
                        prev_fullConvWeights = None

                        # transpose to (OUT_C, K, K, IN_C) then flatten per filter
                        a_tr = np.transpose(a_f, (0, 2, 3, 1))
                        WIN_ELEMS = KH * KW * IN_C
                        a_tr = a_tr.reshape(OUT_C, WIN_ELEMS)
                        a_i32 = float_to_fixed_int32(a_tr, bn_a_frac_bits).astype(np.int32)  # Q12.20
                        b_i32 = float_to_fixed_int32(b_f, bn_b_frac_bits).astype(np.int32)  # Q12.20
                        packed_a = pack_conv_parallel_filters(a_i32, P_FILTER=P_FILTER)  # (DEPTH, P_FILTER)
                        packed_b = pack_bias_parallel_filters(b_i32, P_FILTER=P_FILTER)

                        s = name.replace('running_var', '')
                        s = s.replace('downsample', 'bn')
                        bn_tag = sanitize_tag(s, 7)# + "_" +str(bnCounter))
                        bin_pathA = os.path.join(bin_dir, f"{bn_tag}A.BIN")
                        coe_pathA = os.path.join(coe_dir, f"{bn_tag}A.COE")
                        
                        bin_pathB = os.path.join(bin_dir, f"{bn_tag}B.BIN")
                        coe_pathB = os.path.join(coe_dir, f"{bn_tag}B.COE")

                        csv_pathA = os.path.join(csv_dir, f"{bn_tag}A.CSV")
                        csv_pathB = os.path.join(csv_dir, f"{bn_tag}B.CSV")

                        write_bin_int32(bin_pathA, packed_a.reshape(-1))# packed fused weights. output x kernel x kernel x input
                        write_coe_wide_from_packed_i32(coe_pathA, packed_a, little_endian=little_endian)
                        write_csv_matrix(csv_pathA, packed_a, header=f"{bn_tag}A FUSED packed (DEPTH x P_FILTER) Q{32-bn_a_frac_bits}.{bn_a_frac_bits}")
                        
                        write_bin_int32(bin_pathB, packed_b.reshape(-1))
                        write_coe_wide_from_packed_i32(coe_pathB, packed_b)
                        write_csv_matrix(csv_pathB, packed_b, header=f"{bn_tag}B fused bias Q{32-bn_b_frac_bits}.{bn_b_frac_bits}")

                    else:
                        a_i32 = float_to_fixed_int32(a_f, bn_a_frac_bits).astype(np.int32)  # Q12.20
                        b_i32 = float_to_fixed_int32(b_f, bn_b_frac_bits).astype(np.int32)  # Q12.20
                        words_u64 = pack_bn_ab_bmg(a_i32, b_i32, p_filter=P_FILTER, little_endian=little_endian)

                        s = name.replace('running_var', '')
                        s = s.replace('downsample', 'bn')
                        bn_tag = sanitize_tag(s, 8)# + "_" +str(bnCounter))
                        bin_path = os.path.join(bin_dir, f"{bn_tag}.BIN")
                        coe_path = os.path.join(coe_dir, f"{bn_tag}.COE")
                        csv_path = os.path.join(csv_dir, f"{bn_tag}.CSV")
                        # write_bin_u64(bin_path, words_u64) 
                        write_coe(coe_path, words_u64)
                        # write_csv_bn(csv_path, a_f, b_f, a_i32, b_i32)
                        
                    exported += 1
                    bnCounter += 1
                continue

        # -----------------------------
        # Conv weights (float -> fixed Q12.20, pack for P_FILTER)
        # -----------------------------
        w_np = tensor.detach().numpy()

        if w_np.ndim == 4:
            # PyTorch conv: (OUT_C, IN_C, K, K)
            OUT_C, IN_C, KH, KW = w_np.shape
            if KH != KW:
                raise ValueError(f"Non-square kernel not supported: {name} shape {w_np.shape}")

            # transpose to (OUT_C, K, K, IN_C) then flatten per filter
            w_tr = np.transpose(w_np, (0, 2, 3, 1))
            WIN_ELEMS = KH * KW * IN_C
            w_oc_ke = w_tr.reshape(OUT_C, WIN_ELEMS)

            # fixed point Q12.20
            w_i32 = float_to_fixed_int32(w_oc_ke, conv_frac_bits)

            # pack for parallel filters
            packed = pack_conv_parallel_filters(w_i32, P_FILTER=P_FILTER)  # (DEPTH, P_FILTER)

            conv_tag = sanitize_tag(name)
            bin_path = os.path.join(bin_dir, f"{conv_tag}.BIN")
            coe_path = os.path.join(coe_dir, f"{conv_tag}.COE")
            csv_path = os.path.join(csv_dir, f"{conv_tag}.CSV")

            # BIN: flat int32 stream (DEPTH*P_FILTER)
            write_bin_int32(bin_path, packed.reshape(-1))

            # COE: wide entries (P_FILTER*32 bits)
            write_coe_wide_from_packed_i32(coe_path, packed, little_endian=little_endian)

            # CSV: dump packed matrix (depth rows, P_FILTER cols)
            write_csv_matrix(csv_path, packed, header=f"{name} packed (DEPTH x P_FILTER) Q12.{conv_frac_bits}")

            exported += 1
            prev_fullConvWeights = w_np

            continue

        if w_np.ndim == 2: #FFN Weights:
            # Convert to fixed point
            w_i32 = float_to_fixed_int32(w_np, conv_frac_bits)

            vec_tag = sanitize_tag(name)
            coe_path = os.path.join(coe_dir, f"{vec_tag}.COE")

            # Pack correctly for hardware
            hex_lines = pack_fc_weights_to_coe_lines(
                w_i32,
                parallel_filters=5
            )

            write_coe_from_lines(coe_path, hex_lines)

            # Optional debug outputs
            csv_path = os.path.join(csv_dir, f"{vec_tag}.CSV")
            write_csv_matrix(csv_path, w_i32, header=f"{name} Q{32-conv_frac_bits}.{conv_frac_bits}")

            exported += 1
            continue

        # -----------------------------
        # Other 1D params (non-BN) - optional export
        # -----------------------------
        if w_np.ndim == 1:
            vec_tag = sanitize_tag(name)
            coe_path = os.path.join(coe_dir, f"{vec_tag}.COE")
            csv_path = os.path.join(csv_dir, f"{vec_tag}.CSV")

            OUT_C_FF = len(w_np)
            par = 5
            groups = (OUT_C_FF + par -1) // par
            lines = []
            fixed = float_to_fixed_int32(w_np, conv_frac_bits)

            for g in range(groups):
                lanes = np.zeros(par, dtype=np.uint32)

                for p in range(par):
                    oc = g * par + p
                    if oc < OUT_C_FF:
                        lanes[p] = np.uint32(fixed[oc])

                word = ''.join(f'{x:08X}' for x in lanes[::-1])
                lines.append(word)

            write_coe_from_lines(coe_path, lines)

            # Optional debug outputs
            write_csv_matrix(csv_path, fixed, header=f"{name} Q{32-conv_frac_bits}.{conv_frac_bits}")

            # vec_i32 = float_to_fixed_int32(w_np, conv_frac_bits)
            # vec_tag = sanitize_tag(name)
            # bin_path = os.path.join(bin_dir, f"{vec_tag}.BIN")
            # coe_path = os.path.join(coe_dir, f"{vec_tag}.COE")
            # csv_path = os.path.join(csv_dir, f"{vec_tag}.CSV")

            # write_bin_int32(bin_path, vec_i32)
            # # COE as 32-bit entries
            # write_coe_u32(coe_path, vec_i32.view(np.uint32))
            # write_csv_matrix(csv_path, vec_i32, header=f"{name} Q12.{conv_frac_bits}")

            exported += 1
            continue

        # Otherwise skip (e.g., FC weights 2D) unless you want them too
        # You can add support for 2D (linear layers) if needed.

    print(f"Export complete. Files exported: {exported}")
    print(f"Output dirs:\n  BIN: {bin_dir}\n  COE: {coe_dir}\n  CSV: {csv_dir}")