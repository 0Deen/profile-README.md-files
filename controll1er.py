from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types

from ryu.lib.packet import in_proto, ipv4, icmp, tcp, udp

class Controller(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *_args, **_kwargs):
        super(Controller, self).__init__(*_args, **_kwargs)
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handlaer(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath,
                                    buffer_id=buffer_id, 
                                    priority=priority,
                                    match=match,
                                    instructions=inst
                                    )
        else:
            mod = parser.OFPFlowMod(datapath=datapath, 
                                    priority=priority,
                                    match=match,
                                    instructions=inst
                                    )  
        
        datapath.send_msg(mod)

    def send_packet_out(self, msg, actions):
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=msg.buffer_id,
                                  in_port=in_port,
                                  actions=actions,
                                  data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes", ev.msg.msg_len, ev.msg.total_len)
        
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        dl_dst = eth.dst
        dl_src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.info("packet in %s %s %s %s", dpid, dl_src, dl_dst, in_port)

        self.mac_to_port[dpid][dl_src] = in_port

        if dl_dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dl_dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto

                match = parser.OFPMatch(in_port=in_port,
                                        eth_dst=dl_dst,
                                        eth_src=dl_src)
                
                if protocol == in_proto.IPPROTO_ICMP:
                    icmp_info = pkt.get_protocol(icmp.icmp)
                    print(icmp_info.type)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip,
                                            ipv4_dst=dstip,
                                            eth_src=dl_src,
                                            eth_dst=dl_dst,
                                            in_port=in_port,
                                            ip_proto=protocol)
                    
                elif protocol==in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip,
                                            ipv4_dst=dstip,
                                            eth_src=dl_src,
                                            eth_dst=dl_dst,
                                            in_port=in_port,
                                            ip_proto=protocol,
                                            udp_src=u.src_port,
                                            udp_dst=u.dst_port)
                
                elif protocol==in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    tcp_src = t.src_port
                    tcp_dst = t.dst_port
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                            ipv4_src=srcip,
                                            ipv4_dst=dstip,
                                            eth_src=dl_src,
                                            eth_dst=dl_dst,
                                            in_port=in_port,
                                            ip_proto=protocol,
                                            tcp_src=tcp_src,
                                            tcp_dst=tcp_dst)
                    
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=msg.buffer_id,
                                  in_port=in_port,
                                  actions=actions,
                                  data=data)
        datapath.send_msg(out)  
