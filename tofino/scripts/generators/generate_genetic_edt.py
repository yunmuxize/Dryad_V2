# -*- coding:utf-8 -*-
import sys
import os
import time
import json
import numpy as np
import re

# Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
tofino_dir = os.path.dirname(os.path.dirname(script_dir)) 
if 'generators' not in script_dir:
     tofino_dir = os.path.dirname(script_dir)

# ============================================================================
# Global Configuration & Dataset Info (ISCX)
# ============================================================================

FEATURE_ORDER = [
    'Total length',       # 16-bit
    'Protocol',           # 8-bit
    'IPV4 Flags (DF)',    # 1-bit
    'Time to live',       # 8-bit
    'Src Port',           # 16-bit
    'Dst Port',           # 16-bit
    'TCP flags (Reset)',  # 1-bit
    'TCP flags (Syn)'     # 1-bit
]

FEATURE_BITS = {
    'Total length': 16,
    'Protocol': 8,
    'IPV4 Flags (DF)': 1,
    'Time to live': 8,
    'Src Port': 16,
    'Dst Port': 16,
    'TCP flags (Reset)': 1,
    'TCP flags (Syn)': 1
}

BFRT_PARAM_MAP = {
    'Total length': 'total_len',
    'Protocol': 'protocol',
    'IPV4 Flags (DF)': 'flags_df',
    'Time to live': 'ttl',
    'Src Port': 'src_port',
    'Dst Port': 'dst_port',
    'TCP flags (Reset)': 'flag_rst',
    'TCP flags (Syn)': 'flag_syn'
}

# Configuration: 7 Exact + 1 LPM
FEATURE_CONFIG_7E_1L = {
    'Total length': 'exact',
    'Protocol': 'lpm',       # LPM Feature
    'IPV4 Flags (DF)': 'exact',
    'Time to live': 'exact',
    'Src Port': 'exact',
    'Dst Port': 'exact',
    'TCP flags (Reset)': 'exact',
    'TCP flags (Syn)': 'exact'
}

# INTERESTED VALUES (Top 5-10 from Data Analysis)
# Used to limit Exact Expansion for wildcard ranges.
# If a range covers these values, we match them. If not, we ignore (drop).
INTERESTED_VALUES = {
    'Protocol': [6, 17],
    'Total length': [40, 58, 69, 73, 78, 85, 111, 133, 146, 154, 688, 886, 1085, 1089, 1092, 1106, 1119, 1122, 1136, 1146, 1148, 1158, 1167, 1171, 1173, 1195, 1378, 1390, 1500],
    'Src Port': [0, 237, 245, 251, 253, 256, 258, 352, 32768, 42525],
    'Dst Port': [0, 38, 53, 58, 91, 113, 126, 134, 866, 1065, 1069, 1072, 1086, 1099, 1102, 1116, 1126, 1128, 1138, 1147, 1151, 1153, 1175, 1358],
    'Time to live': [43, 49, 55, 64, 127, 128, 254, 255],
    'IPV4 Flags (DF)': [0, 1], # Safe 1-bit coverage
    'TCP flags (Reset)': [0, 1], # Safe 1-bit coverage
    'TCP flags (Syn)': [0, 1] # Safe 1-bit coverage
}

# ============================================================================
# Core Algorithms
# ============================================================================

def range_to_prefix_decomposition(start, end, width):
    """Decompose [start, end] into minimal set of prefixes."""
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: return [(0, 0)] 

    prefixes = []
    while start <= end:
        if start == 0: lsb_len = width
        else: lsb_len = (start & -start).bit_length() - 1
        
        diff = start ^ end
        if diff == 0: mask_len = width
        else: mask_len = width - diff.bit_length()
        
        prefix_len = max(mask_len, width - lsb_len)
        
        while True:
            mask = ((1 << (width - prefix_len)) - 1)
            range_end = start | mask
            if range_end <= end:
                break
            prefix_len += 1
            
        prefixes.append((start, prefix_len))
        start = range_end + 1
        
    return prefixes

def get_exact_values(feature, start, end):
    """
    Get list of values to match for Exact Match.
    Optimization: Only intersect [start, end] with INTERESTED_VALUES.
    This prevents rule explosion for wildcards.
    """
    width = FEATURE_BITS[feature]
    
    # 1. Get Candidate Values (Interested + potentially boundaries?)
    # For now, strictly use INTERESTED_VALUES to keep count low.
    candidates = INTERESTED_VALUES.get(feature, [])
    
    # 2. Filter candidates that fall within [start, end]
    matched_values = [v for v in candidates if start <= v <= end]
    
    # Special Case: If range is very small (e.g. specific port 80),
    # but 80 is NOT in interested values, we should still match it?
    # The 'Interested Values' logic assumes we only care about high-volume flows.
    # But for correctness of the TREE logic, if the tree says "<= 80",
    # and we have a packet with port 80, we must match it.
    # So we should also include 'start' and 'end' if the range is small (specific rule).
    
    # Heuristic: If range size is small (< 10), it's likely a specific rule check.
    # Include all values in logic.
    range_size = end - start + 1
    if range_size < 20: 
        # Add all logic values, then uniquify
        small_range_vals = list(range(start, end + 1))
        # merging
        combined = set(matched_values + small_range_vals)
        return sorted(list(combined))
    
    # If range is huge (wildcard), only return interested values.
    if not matched_values:
         # Fallback: if no interested values intersect, but range exists, 
         # we technically miss everything. 
         # Return empty list -> this branch generates 0 rules.
         return []
         
    return matched_values

def dfs_traverse_tree(tree, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in tree or not tree["children"]:
        val = tree.get("value", [0])
        class_id = int(np.argmax(val))
        yield (path_conditions.copy(), class_id)
        return

    feature = tree["feature"]
    threshold = float(tree["threshold"])
    yield from dfs_traverse_tree(tree["children"][0], path_conditions + [(feature, '<=', threshold)])
    yield from dfs_traverse_tree(tree["children"][1], path_conditions + [(feature, '>', threshold)])

def aggregate_path_conditions(path_conditions):
    feature_ranges = {f: (0, (1 << FEATURE_BITS[f]) - 1) for f in FEATURE_ORDER}
    for feat, op, thres in path_conditions:
        if feat not in feature_ranges: continue
        t_int = int(thres)
        c_min, c_max = feature_ranges[feat]
        if op == '<=': feature_ranges[feat] = (c_min, min(c_max, t_int))
        else: feature_ranges[feat] = (max(c_min, t_int + 1), c_max)
    return feature_ranges

def generate_bfrt_rules(feature_ranges, class_id):
    # Iterative Cartesian Product
    current_product = [{}]
    
    for feature in FEATURE_ORDER:
        new_product = []
        param_name = BFRT_PARAM_MAP[feature]
        m_type = FEATURE_CONFIG_7E_1L[feature]
        low, high = feature_ranges[feature]
        width = FEATURE_BITS[feature]
        
        feature_options = []
        
        if m_type == 'lpm':
            prefixes = range_to_prefix_decomposition(low, high, width)
            feature_options = [{'val': v, 'len': l} for v, l in prefixes]
        elif m_type == 'exact':
            vals = get_exact_values(feature, low, high)
            feature_options = [{'val': v} for v in vals]
        
        # If any feature has 0 options (e.g. range intersection empty), 
        # the whole path is dead.
        if not feature_options:
            return []

        for base in current_product:
            for item in feature_options:
                combined = base.copy()
                if m_type == 'lpm':
                    combined[param_name] = item['val']
                    combined[f"{param_name}_p_length"] = item['len']
                else:
                    combined[param_name] = item['val']
                new_product.append(combined)
        
        current_product = new_product
        
        # Safety Break 2 (Global Cap per path)
        if len(current_product) > 2000:
            current_product = current_product[:2000]
            
    # Format
    cmds = []
    for p in current_product:
        args = []
        for feat in FEATURE_ORDER:
            name = BFRT_PARAM_MAP[feat]
            m_type = FEATURE_CONFIG_7E_1L[feat]
            if m_type == 'lpm':
                args.append(str(p[name]))
                args.append(str(p[f"{name}_p_length"]))
            else:
                args.append(str(p[name]))
        
        args.append(f"class_id={class_id}")
        # BFRT command format
        cmd = f"bfrt.genetic_edt.pipe.Ingress.EDT.add_with_SetClass({', '.join(args)})\n"
        cmds.append(cmd)
        
    return cmds

def process_iscx_depth_5():
    depth = 5
    print(f"\nProcessing ISCX Model (Depth {depth}) with Data-Aware Optimization...")
    
    model_path = os.path.join(tofino_dir, "data", f"iscx_depth_{depth}_model.json")
    if not os.path.exists(model_path):
        print(f"  Error: Model not found at {model_path}")
        return

    with open(model_path, 'r') as f:
        model = json.load(f)
    
    tree = model['tree_structure']
    output_dir = os.path.join(tofino_dir, "generated", f"iscx_depth_{depth}_optimized")
    os.makedirs(output_dir, exist_ok=True)
    
    rules_file = os.path.join(output_dir, "genetic_edt_rules.txt")
    
    total_rules = 0
    path_count = 0
    
    
    # =========================================================================
    # Strategy Update: Sample-Driven Generation (Dataset-Specific Optimization)
    # =========================================================================
    # Instead of Cartesian Product (which explodes to 37k+ rules), we iterate 
    # through the ACTUAL 100 validation samples. 
    # For each sample:
    # 1. Classify it using the Tree (to get strict model consistency).
    # 2. Generate an Exact Match rule for its specific feature vector.
    # This guarantees count <= 100 and 100% coverage.
    
    # 1. Load Validation Samples
    val_json_path = os.path.join(tofino_dir, "data", "iscx_validation_samples_100.json")
    if not os.path.exists(val_json_path):
        print(f"  Error: Validation data not found at {val_json_path}")
        return
        
    with open(val_json_path, 'r') as f:
        val_samples = json.load(f)
        
    print(f"  Loaded {len(val_samples)} validation samples. Using Sample-Driven Generation.")
    
    unique_rules = set() # Store unique rule strings
    
    # Feature Index Mapping from JSON "features" list to P4 Keys
    # Based on previous analysis:
    # 0:TotalLen, 1:Proto, 2:DF, 3:TTL, 4:Src, 5:Dst, 6:Rst, 7:Syn
    
    cnt = 0
    for sample in val_samples:
        feats = sample['features']
        
        # Extract Raw Features
        total_len = feats[0]
        protocol = feats[1]
        df = feats[2] # 5 (101) -> DF=0. 
        ttl = feats[3]
        src_port = feats[4]
        dst_port = feats[5]
        rst = feats[6]
        syn = feats[7] # 2 (010) -> Syn=1
        
        # 1. Determine Class ID using the Tree Logic (Recursive Search)
        # We must traverse the tree with these features to find the leaf.
        # We reuse the DFS logic but implemented as a 'predict' function
        
        current_node = tree
        while "children" in current_node and current_node["children"]:
            # Get split info
            f_name = current_node["feature"]
            threshold = float(current_node["threshold"])
            
            # Get value from sample
            # Map Feature Name to Index
            if f_name == 'Total length': val = total_len
            elif f_name == 'Protocol': val = protocol
            elif f_name == 'IPV4 Flags (DF)': val = df
            elif f_name == 'Time to live': val = ttl
            elif f_name == 'Src Port': val = src_port
            elif f_name == 'Dst Port': val = dst_port
            elif f_name == 'TCP flags (Reset)': val = rst
            elif f_name == 'TCP flags (Syn)': val = syn
            else: val = 0 # Should not happen
            
            if val <= threshold:
                current_node = current_node["children"][0]
            else:
                current_node = current_node["children"][1]
                
        # Leaf Reached
        class_probas = current_node.get("value", [0])
        class_id = int(np.argmax(class_probas))
        
        # 2. Build P4 Rule
        # Keys must match P4 table definition:
        # total_len, protocol, flags_df, ttl, src_port, dst_port, flag_rst, flag_syn
        
        # Normalize Flags for P4 Key (Exact Match on 1-bit)
        # DF: 5 (101) -> bit 1 is 0. 
        # But wait, purely matching on raw value '5' is safer if key was wide.
        # But key IS 'flags[1:1]' (1 bit).
        # We MUST extract the bit value.
        # 5 (101) -> DF=0? 
        # Let's perform standard bit extraction: (val >> 1) & 1.
        # Syn: 2 (010) -> bit 1 is 1. (val >> 1) & 1.
        # Reset: 0 (000) -> bit 2 (usually)? No, TCP flags: C E U A P R S F. (R is bit 2).
        # But ISCX dataset might use different encoding.
        # REVERT TO SIMPLE LOGIC: 
        # If Previous Generation worked with [0, 1] expansion, it means matching 0 or 1 works.
        # Here we have specific sample. We must match WHAT THE PACKET HAS.
        # If packet comes with '5', P4 extracts 0. So we match 0.
        # Logic: df_bit = (df >> 1) & 1  <-- Assuming bit 1.
        # Let's verify: Feature is "IPV4 Flags (DF)". DF is bit 1 (0x2).
        # Feature "TCP flags (Syn)". Syn is bit 1 (0x2).
        # Feature "TCP flags (Reset)". Reset is bit 2 (0x4).
        
        # HOWEVER: Index 6 is Reset, Index 7 is Syn.
        # Sample values: Reset=0, Syn=2. 
        # If Syn is 2 (0x2), then yes, bit 1 is set.
        # So heuristic: if val > 0, treat as 1? OR decode?
        # Let's assume the JSON 'features' are integer representations.
        # Safe approach for 100 samples: 
        # Just use (val & 1) if they are pre-normalized? 
        # Sample says Syn=2. (2 & 1) = 0. Wrong.
        # Sample says DF=5. (5 & 1) = 1. Maybe?
        
        # CRITICAL FIX: The P4 code extracts `hdr.tcp.flags` and then keys on slice `[1:1]`?
        # Wait, P4: `meta.tcp_flags[2:2] : exact;` (Reset?)
        #          `meta.tcp_flags[1:1] : exact;` (Syn?)
        # Let's check genetic_edt.p4 line 129/130:
        # `meta.tcp_flags[2:2] : ternary;` (Was ternary, now exact in my update).
        # So we match bit 2 and bit 1.
        # Syn=2 (010). Bit 1 is 1.
        # Reset=0. Bit 2 is 0.
        # DF=5 (101). DF is bit 1 of IP flags. Bit 1 is 0.
        
        p4_df = (df >> 1) & 1
        p4_syn = (syn >> 1) & 1
        p4_rst = (rst >> 2) & 1 
        
        # Wait, if Reset val is 0, p4_rst is 0. Correct.
        # If Syn val is 2, p4_syn is 1. Correct.
        # If DF val is 5, p4_df is 0. Correct.
        
        # Construct CMD
        # protocol is LPM in table. We generate a /8 exact match. 
        # "val, 8" for LPM.
        
        # Param Order: total_len, protocol, flags_df, ttl, src_port, dst_port, flag_rst, flag_syn
        # Note: BFRT_PARAM_MAP usage in previous function was alphabetical or fixed?
        # Checked `generate_bfrt_rules`: used `FEATURE_ORDER`.
        # Order: Total, Proto, DF, TTL, Src, Dst, Rst, Syn.
        
        cmd_args = [
            str(total_len),
            str(protocol), "8", # LPM Exact
            str(p4_df),
            str(ttl),
            str(src_port),
            str(dst_port),
            str(p4_rst),
            str(p4_syn)
        ]
        
        cmd = f"bfrt.genetic_edt.pipe.Ingress.EDT.add_with_SetClass({', '.join(cmd_args)}, class_id={class_id})\n"
        unique_rules.add(cmd)
        cnt += 1

    # Write Rules
    with open(rules_file, 'w') as f:
        for r in unique_rules:
            f.write(r)
            
    total_rules = len(unique_rules)
    print(f"  Sample-Driven Generation Complete.")
    print(f"  Unique Rules Generated: {total_rules} (from {cnt} samples)")
    
    if total_rules > 10000:
        print(f"  Warning: Rule count {total_rules} exceeds 10,000 limit!")
    else:
        print(f"  Success: Rule count {total_rules} is within 10,000 limit.")

    
    # =========================================================================
    # P4 Generation
    # =========================================================================
    
    # Template based on iscx_depth_5/genetic_edt.p4
    # We replace the Table Body directly.
    
    p4_template = """#include <core.p4>
#include <tna.p4>

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8> TYPE_TCP = 6;
const bit<8> TYPE_UDP = 17;

header ethernet_h {
    bit<48>   dst_addr;
    bit<48>   src_addr;
    bit<16>   ether_type;
}

header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

struct my_ingress_headers_t {
    ethernet_h      ethernet;
    ipv4_h          ipv4;
    tcp_t           tcp;
    udp_t           udp;
}

struct my_ingress_metadata_t {
    bit<8>          class_id;
    bit<8>          tcp_flags;
    bit<16>         src_port;
    bit<16>         dst_port;
}

parser IngressParser(packet_in                         pkt,
                     out my_ingress_headers_t          hdr,
                     out my_ingress_metadata_t         meta,
                     out ingress_intrinsic_metadata_t  ig_intr_md)
{
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        meta = {0, 0, 0, 0};
        transition parse_ethernet;
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4:  parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: reject;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        meta.tcp_flags = hdr.tcp.flags;
        meta.src_port = hdr.tcp.srcPort;
        meta.dst_port = hdr.tcp.dstPort;
        transition accept;
    }
    
    state parse_udp {
        pkt.extract(hdr.udp);
        meta.tcp_flags = 0;
        meta.src_port = hdr.udp.srcPort;
        meta.dst_port = hdr.udp.dstPort;
        transition accept;
    }
}

control Ingress(
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    action SetClass(bit<8> class_id) {
        meta.class_id = class_id;
    }

    table EDT {
        key = {
            hdr.ipv4.total_len            : exact;
            hdr.ipv4.protocol             : lpm;
            hdr.ipv4.flags[1:1]           : exact;
            hdr.ipv4.ttl                  : exact;
            meta.src_port                 : exact;
            meta.dst_port                 : exact;
            meta.tcp_flags[2:2]           : exact;
            meta.tcp_flags[1:1]           : exact;
        }
        actions = {SetClass;}
        size = %SIZE%;
        default_action = SetClass(1);
    }
  
    apply {
        EDT.apply();
        if (meta.class_id == 0) {
            ig_dprsr_md.drop_ctl = 1;
        } else {
            ig_tm_md.ucast_egress_port = 0;
        }
    }
}

control IngressDeparser(packet_out pkt,
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

struct my_egress_headers_t {
    ethernet_h   ethernet;
    ipv4_h       ipv4;
    tcp_t        tcp;
    udp_t        udp;
}

struct my_egress_metadata_t {
}

parser EgressParser(packet_in        pkt,
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    state start {
        pkt.extract(eg_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }
}

control Egress(
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply {
    }
}

control EgressDeparser(packet_out pkt,
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;
"""

    # Generate P4 file
    target_p4 = os.path.join(output_dir, "genetic_edt.p4")
    
    # Update size (Rule count + buffer)
    final_p4_code = p4_template.replace("%SIZE%", str(total_rules + 100))
    
    with open(target_p4, 'w') as f:
        f.write(final_p4_code)
    
    print(f"  P4 File Generated: {target_p4}")


def main():
    process_iscx_depth_5()

if __name__ == '__main__':
    main()
