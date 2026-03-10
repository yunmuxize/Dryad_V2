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
    bit<1> codes_f0;
	bit<2> codes_f2;
	bit<1> codes_f3;
	bit<40> codes_f4;
	bit<4> codes_f5;
	bit<7> codes_f6;
	bit<21> codes_f7;
	bit<28> codes_f8;
	bit<36> codes_f9;
	
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
            meta.codes_f0 : ternary;
		meta.codes_f2 : ternary;
		meta.codes_f3 : ternary;
		meta.codes_f4 : ternary;
		meta.codes_f5 : ternary;
		meta.codes_f6 : ternary;
		meta.codes_f7 : ternary;
		meta.codes_f8 : ternary;
		meta.codes_f9 : ternary;
		
        }
        actions = {
            ac_packet_forward;
            default_forward;
        }
        default_action = default_forward();
        size=24780;
    }
    action ac_fea_f0(bit<1> code){
		meta.codes_f0 = code;
	}

	table tbl_fea_f0{
		key= {hdr.ipv4.protocol : ternary;}
		actions = {ac_fea_f0;}
		size=2;
	}

	action ac_fea_f2(bit<2> code){
		meta.codes_f2 = code;
	}

	table tbl_fea_f2{
		key= {hdr.ipv4.tos : ternary;}
		actions = {ac_fea_f2;}
		size=2;
	}

	action ac_fea_f3(bit<1> code){
		meta.codes_f3 = code;
	}

	table tbl_fea_f3{
		key= {hdr.ipv4.flags : ternary;}
		actions = {ac_fea_f3;}
		size=1;
	}

	action ac_fea_f4(bit<40> code){
		meta.codes_f4 = code;
	}

	table tbl_fea_f4{
		key= {hdr.ipv4.ttl : ternary;}
		actions = {ac_fea_f4;}
		size=70;
	}

	action ac_fea_f5(bit<4> code){
		meta.codes_f5 = code;
	}

	table tbl_fea_f5{
		key= {meta.dataOffset : ternary;}
		actions = {ac_fea_f5;}
		size=6;
	}

	action ac_fea_f6(bit<7> code){
		meta.codes_f6 = code;
	}

	table tbl_fea_f6{
		key= {meta.flags : ternary;}
		actions = {ac_fea_f6;}
		size=10;
	}

	action ac_fea_f7(bit<21> code){
		meta.codes_f7 = code;
	}

	table tbl_fea_f7{
		key= {meta.window : ternary;}
		actions = {ac_fea_f7;}
		size=71;
	}

	action ac_fea_f8(bit<28> code){
		meta.codes_f8 = code;
	}

	table tbl_fea_f8{
		key= {meta.udp_length : ternary;}
		actions = {ac_fea_f8;}
		size=70;
	}

	action ac_fea_f9(bit<36> code){
		meta.codes_f9 = code;
	}

	table tbl_fea_f9{
		key= {hdr.ipv4.totalLen : ternary;}
		actions = {ac_fea_f9;}
		size=92;
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

