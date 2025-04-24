from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
import random
import time

# Function to simulate DDoS traffic
def simulate_ddos(net, target_ip, duration=60):
    print(f"Simulating DDoS attack on {target_ip} for {duration} seconds")
    attackers = []
    for host in net.hosts:
        if host.IP() != target_ip:
            attackers.append(host)
    
    # Simulate SYN flood by sending TCP packets
    for attacker in attackers:
        attacker.cmd(f"sudo hping3 -S {target_ip} -p 80 --flood &")

    # Attack lasts for 'duration' seconds
    time.sleep(duration)

    # Stop the attack after the specified duration
    for attacker in attackers:
        attacker.cmd("sudo killall hping3")
    print(f"DDoS simulation completed for {target_ip}")

# Set up Mininet network topology
def create_network():
    net = Mininet(controller=Controller, switch=OVSSwitch, link=TCLink)
    
    # Create a controller
    c0 = net.addController('c0')
    
    # Add hosts and switches
    h1 = net.addHost('h1', ip="10.0.0.1")
    h2 = net.addHost('h2', ip="10.0.0.2")
    h3 = net.addHost('h3', ip="10.0.0.3")
    
    s1 = net.addSwitch('s1')
    
    # Link the devices together
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)
    
    # Start the network
    net.start()
    
    # Simulate attack on h2
    simulate_ddos(net, "10.0.0.2", duration=30)
    
    # Start the CLI
    CLI(net)
    net.stop()

if __name__ == '__main__':
    create_network()
