 /* -*- P4_16 -*- */
#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
#include "headers.p4"
#include "egress.p4"

const bit<16> TYPE_IPV4 = 0x800;
const bit<8> PROTO_TCP = 6;
const bit<8> PROTO_UDP = 17;

struct my_ingress_metadata_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> udp_length;
    bit<4>  dataOffset;
    bit<16> window;
    bit<8>  flags;
    bit<1> codes_0_f0;
	bit<1> codes_1_f0;
	bit<1> codes_4_f0;
	bit<1> codes_4_f2;
	bit<2> codes_0_f3;
	bit<2> codes_1_f3;
	bit<1> codes_2_f3;
	bit<2> codes_3_f3;
	bit<1> codes_4_f3;
	bit<1> codes_5_f3;
	bit<1> codes_6_f3;
	bit<3> codes_0_f4;
	bit<5> codes_1_f4;
	bit<2> codes_2_f4;
	bit<4> codes_3_f4;
	bit<3> codes_4_f4;
	bit<3> codes_5_f4;
	bit<4> codes_6_f4;
	bit<1> codes_0_f5;
	bit<1> codes_1_f5;
	bit<2> codes_2_f5;
	bit<2> codes_3_f5;
	bit<2> codes_4_f5;
	bit<1> codes_5_f5;
	bit<1> codes_0_f6;
	bit<3> codes_1_f6;
	bit<1> codes_2_f6;
	bit<1> codes_3_f6;
	bit<2> codes_4_f6;
	bit<1> codes_5_f6;
	bit<2> codes_6_f6;
	bit<2> codes_0_f7;
	bit<1> codes_1_f7;
	bit<4> codes_2_f7;
	bit<1> codes_3_f7;
	bit<4> codes_4_f7;
	bit<1> codes_5_f7;
	bit<2> codes_0_f8;
	bit<1> codes_2_f8;
	bit<2> codes_3_f8;
	bit<1> codes_5_f8;
	bit<4> codes_6_f8;
	bit<3> codes_0_f9;
	bit<2> codes_1_f9;
	bit<4> codes_2_f9;
	bit<3> codes_3_f9;
	bit<1> codes_4_f9;
	bit<7> codes_5_f9;
	bit<4> codes_6_f9;
	bit<3> pred_0;
	bit<3> pred_1;
	bit<3> pred_2;
	bit<3> pred_3;
	bit<3> pred_4;
	bit<3> pred_5;
	bit<3> pred_6;
	
}

struct my_ingress_headers_t {
    ethernet_t  ethernet;
    ipv4_t      ipv4;
    tcp_t       tcp;
    udp_t       udp;
}


parser IngressParser(packet_in        pkt,
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    out ingress_intrinsic_metadata_t  ig_intr_md)
{

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition parse_ipv4;
        }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP   : parse_tcp;
            PROTO_UDP   : parse_udp;
            // default: accept;
        }
   }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        meta.dataOffset = hdr.tcp.dataOffset;
        meta.window = hdr.tcp.window;
        meta.flags = hdr.tcp.flags;
        meta.udp_length = 0x0;
        meta.srcPort=hdr.tcp.srcPort;
        meta.dstPort=hdr.tcp.dstPort;
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.dataOffset = 0x0;
        meta.window = 0x0;
        meta.flags = 0x0;
        meta.udp_length = hdr.udp.udp_length;
        meta.srcPort=hdr.udp.srcPort;
        meta.dstPort=hdr.udp.dstPort;
        transition accept;
    }
}


control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

    action ac_packet_forward(PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
#ifdef BYPASS_EGRESS
        ig_tm_md.bypass_egress = 1;
#endif
    }

    action default_forward() {
        ig_tm_md.ucast_egress_port = 2;
#ifdef BYPASS_EGRESS
        ig_tm_md.bypass_egress = 1;
#endif
    }

    table tb_packet_cls {
        key = {
            meta.pred_0 : exact;
		meta.pred_1 : exact;
		meta.pred_2 : exact;
		meta.pred_3 : exact;
		meta.pred_4 : exact;
		meta.pred_5 : exact;
		meta.pred_6 : exact;
		
        }
        actions = {
            ac_packet_forward;
            default_forward;
        }
        default_action = default_forward();
        size=135000;
    }
    action ac_fea_f0(bit<1> code0, bit<1> code1, bit<1> code4){
		meta.codes_0_f0 = code0;
		meta.codes_1_f0 = code1;
		meta.codes_4_f0 = code4;
	}

	table tbl_fea_f0{
		key= {meta.srcPort : range;}
		actions = {ac_fea_f0;}
		size=2;
	}

	action ac_fea_f2(bit<1> code4){
		meta.codes_4_f2 = code4;
	}

	table tbl_fea_f2{
		key= {hdr.ipv4.protocol : range;}
		actions = {ac_fea_f2;}
		size=2;
	}

	action ac_fea_f3(bit<2> code0, bit<2> code1, bit<1> code2, bit<2> code3, bit<1> code4, bit<1> code5, bit<1> code6){
		meta.codes_0_f3 = code0;
		meta.codes_1_f3 = code1;
		meta.codes_2_f3 = code2;
		meta.codes_3_f3 = code3;
		meta.codes_4_f3 = code4;
		meta.codes_5_f3 = code5;
		meta.codes_6_f3 = code6;
	}

	table tbl_fea_f3{
		key= {hdr.ipv4.ihl : range;}
		actions = {ac_fea_f3;}
		size=2;
	}

	action ac_fea_f4(bit<3> code0, bit<5> code1, bit<2> code2, bit<4> code3, bit<3> code4, bit<3> code5, bit<4> code6){
		meta.codes_0_f4 = code0;
		meta.codes_1_f4 = code1;
		meta.codes_2_f4 = code2;
		meta.codes_3_f4 = code3;
		meta.codes_4_f4 = code4;
		meta.codes_5_f4 = code5;
		meta.codes_6_f4 = code6;
	}

	table tbl_fea_f4{
		key= {hdr.ipv4.tos : range;}
		actions = {ac_fea_f4;}
		size=18;
	}

	action ac_fea_f5(bit<1> code0, bit<1> code1, bit<2> code2, bit<2> code3, bit<2> code4, bit<1> code5){
		meta.codes_0_f5 = code0;
		meta.codes_1_f5 = code1;
		meta.codes_2_f5 = code2;
		meta.codes_3_f5 = code3;
		meta.codes_4_f5 = code4;
		meta.codes_5_f5 = code5;
	}

	table tbl_fea_f5{
		key= {hdr.ipv4.ttl : range;}
		actions = {ac_fea_f5;}
		size=4;
	}

	action ac_fea_f6(bit<1> code0, bit<3> code1, bit<1> code2, bit<1> code3, bit<2> code4, bit<1> code5, bit<2> code6){
		meta.codes_0_f6 = code0;
		meta.codes_1_f6 = code1;
		meta.codes_2_f6 = code2;
		meta.codes_3_f6 = code3;
		meta.codes_4_f6 = code4;
		meta.codes_5_f6 = code5;
		meta.codes_6_f6 = code6;
	}

	table tbl_fea_f6{
		key= {meta.dataOffset : range;}
		actions = {ac_fea_f6;}
		size=4;
	}

	action ac_fea_f7(bit<2> code0, bit<1> code1, bit<4> code2, bit<1> code3, bit<4> code4, bit<1> code5){
		meta.codes_0_f7 = code0;
		meta.codes_1_f7 = code1;
		meta.codes_2_f7 = code2;
		meta.codes_3_f7 = code3;
		meta.codes_4_f7 = code4;
		meta.codes_5_f7 = code5;
	}

	table tbl_fea_f7{
		key= {meta.window : range;}
		actions = {ac_fea_f7;}
		size=7;
	}

	action ac_fea_f8(bit<2> code0, bit<1> code2, bit<2> code3, bit<1> code5, bit<4> code6){
		meta.codes_0_f8 = code0;
		meta.codes_2_f8 = code2;
		meta.codes_3_f8 = code3;
		meta.codes_5_f8 = code5;
		meta.codes_6_f8 = code6;
	}

	table tbl_fea_f8{
		key= {meta.udp_length : range;}
		actions = {ac_fea_f8;}
		size=9;
	}

	action ac_fea_f9(bit<3> code0, bit<2> code1, bit<4> code2, bit<3> code3, bit<1> code4, bit<7> code5, bit<4> code6){
		meta.codes_0_f9 = code0;
		meta.codes_1_f9 = code1;
		meta.codes_2_f9 = code2;
		meta.codes_3_f9 = code3;
		meta.codes_4_f9 = code4;
		meta.codes_5_f9 = code5;
		meta.codes_6_f9 = code6;
	}

	table tbl_fea_f9{
		key= {hdr.ipv4.totalLen : range;}
		actions = {ac_fea_f9;}
		size=21;
	}

	
    action ac_tree_0(bit<3> cls) {
		meta.pred_0 = cls;
	}

	table tbl_tree_0{
		key={
meta.codes_0_f0 : ternary;
		meta.codes_0_f3 : ternary;
		meta.codes_0_f4 : ternary;
		meta.codes_0_f5 : ternary;
		meta.codes_0_f6 : ternary;
		meta.codes_0_f7 : ternary;
		meta.codes_0_f8 : ternary;
		meta.codes_0_f9 : ternary;
		}
		actions = {ac_tree_0;}
		size=16;
	}

	action ac_tree_1(bit<3> cls) {
		meta.pred_1 = cls;
	}

	table tbl_tree_1{
		key={
meta.codes_1_f0 : ternary;
		meta.codes_1_f3 : ternary;
		meta.codes_1_f4 : ternary;
		meta.codes_1_f5 : ternary;
		meta.codes_1_f6 : ternary;
		meta.codes_1_f7 : ternary;
		meta.codes_1_f9 : ternary;
		}
		actions = {ac_tree_1;}
		size=16;
	}

	action ac_tree_2(bit<3> cls) {
		meta.pred_2 = cls;
	}

	table tbl_tree_2{
		key={
meta.codes_2_f3 : ternary;
		meta.codes_2_f4 : ternary;
		meta.codes_2_f5 : ternary;
		meta.codes_2_f6 : ternary;
		meta.codes_2_f7 : ternary;
		meta.codes_2_f8 : ternary;
		meta.codes_2_f9 : ternary;
		}
		actions = {ac_tree_2;}
		size=16;
	}

	action ac_tree_3(bit<3> cls) {
		meta.pred_3 = cls;
	}

	table tbl_tree_3{
		key={
meta.codes_3_f3 : ternary;
		meta.codes_3_f4 : ternary;
		meta.codes_3_f5 : ternary;
		meta.codes_3_f6 : ternary;
		meta.codes_3_f7 : ternary;
		meta.codes_3_f8 : ternary;
		meta.codes_3_f9 : ternary;
		}
		actions = {ac_tree_3;}
		size=16;
	}

	action ac_tree_4(bit<3> cls) {
		meta.pred_4 = cls;
	}

	table tbl_tree_4{
		key={
meta.codes_4_f0 : ternary;
		meta.codes_4_f2 : ternary;
		meta.codes_4_f3 : ternary;
		meta.codes_4_f4 : ternary;
		meta.codes_4_f5 : ternary;
		meta.codes_4_f6 : ternary;
		meta.codes_4_f7 : ternary;
		meta.codes_4_f9 : ternary;
		}
		actions = {ac_tree_4;}
		size=16;
	}

	action ac_tree_5(bit<3> cls) {
		meta.pred_5 = cls;
	}

	table tbl_tree_5{
		key={
meta.codes_5_f3 : ternary;
		meta.codes_5_f4 : ternary;
		meta.codes_5_f5 : ternary;
		meta.codes_5_f6 : ternary;
		meta.codes_5_f7 : ternary;
		meta.codes_5_f8 : ternary;
		meta.codes_5_f9 : ternary;
		}
		actions = {ac_tree_5;}
		size=16;
	}

	action ac_tree_6(bit<3> cls) {
		meta.pred_6 = cls;
	}

	table tbl_tree_6{
		key={
meta.codes_6_f3 : ternary;
		meta.codes_6_f4 : ternary;
		meta.codes_6_f6 : ternary;
		meta.codes_6_f8 : ternary;
		meta.codes_6_f9 : ternary;
		}
		actions = {ac_tree_6;}
		size=16;
	}

	

    apply {
        tbl_fea_f0.apply();
		tbl_fea_f2.apply();
		tbl_fea_f3.apply();
		tbl_fea_f4.apply();
		tbl_fea_f5.apply();
		tbl_fea_f6.apply();
		tbl_fea_f7.apply();
		tbl_fea_f8.apply();
		tbl_fea_f9.apply();
		tbl_tree_0.apply();
		tbl_tree_1.apply();
		tbl_tree_2.apply();
		tbl_tree_3.apply();
		tbl_tree_4.apply();
		tbl_tree_5.apply();
		tbl_tree_6.apply();
		
        tb_packet_cls.apply();
    }

}

control IngressDeparser(
    packet_out pkt,
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
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

