import numpy as np


def load_coe_lines(filename):
    """Read COE file and return list of 256-bit hex strings."""
    lines = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("memory") or line == "":
                continue

            line = line.replace(",", "")
            lines.append(line)

    return lines


def extract_filter0_from_coe(coe_lines, Cin=64, K=3, parallel_filters=8):
    """
    Reconstruct 3x3x64 weight tensor for filter 0
    """

    cin_groups = Cin // 32
    ACT_WORDS = K * K * cin_groups

    weights = np.zeros((Cin, K, K), dtype=np.int8)

    addr = 0

    for ky in range(K):
        for kx in range(K):
            for cg in range(cin_groups):

                line = coe_lines[addr]

                # split 256-bit word into 8 lanes
                lanes = [int(line[i:i+8], 16) for i in range(0, 64, 8)]

                # lane0 = filter0
                lane0 = lanes[-1]  # because hex string is big-endian

                for b in range(32):
                    c = cg*32 + b
                    bit = (lane0 >> b) & 1
                    weights[c, ky, kx] = bit

                addr += 1

    return weights


# Example usage
coe_lines = load_coe_lines("pytorch_implementation\BiReal18_34\savedWeights\WEIGHTS\COE\\10BINARY.COE")

filter0 = extract_filter0_from_coe(coe_lines)

print("Filter 0 shape:", filter0.shape)
print(filter0)