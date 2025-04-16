#!/usr/bin/python
import threading
import random
import time
from mininet.log import setLogLevel, info
from mininet.topo import Topo
from mininet.net import Mininet, CLI
from mininet.node import OVSKernelSwitch, Host
from mininet.link import TCLink, Link
from mininet.node import RemoteController #Controller

import time

class Environment(object):
    def __init__(self):
        "Create a network."
        self.net = Mininet(controller=RemoteController, link=TCLink)
        info("*** Starting controller\n")

        c1 = self.net.addController( 'c1', controller=RemoteController, ip = '127.0.0.1', protocol='tcp', port=6633) #Controller
        c1.start()
        
        info("*** Adding hosts\n")
        self.h1 = self.net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1')  #host malware
        self.h2 = self.net.addHost('h2', mac='00:00:00:00:00:02', ip='10.0.0.2')
        self.h3 = self.net.addHost('h3', mac='00:00:00:00:00:03', ip='10.0.0.3')

        info("*** Adding switches\n")
        self.s1, self.s2, self.s3, self.s4 = [ self.net.addSwitch(s, failMode="standalone") for s in ('s1', 's2', 's3', 's4')]

        info("*** Adding links\n")  
        for h, s in [(self.h1, self.s1), (self.h2,self.s2), (self.h3, self.s4)]:
            self.net.addLink(h, s)
        
        self.net.addLink(self.s1, self.s3)
        self.net.addLink(self.s2, self.s3)
        self.net.addLink(self.s3, self.s4)

        info("*** Starting network\n")
        self.net.build()
        self.net.start()

...
if __name__ == '__main__':

    setLogLevel('info')
    info('starting the environment\n')
    env = Environment()

    info("*** Running CLI\n")
    CLI(env.net)
    setLogLevel('info')

    env.net.stop()
