# Prefix to Subnet mask
P2S = {
"0": "0.0.0.0",
"1": "128.0.0.0",
"2": "192.0.0.0",
"3": "224.0.0.0",
"4": "240.0.0.0",
"5": "248.0.0.0",
"6": "252.0.0.0",
"7": "254.0.0.0",
"8": "255.0.0.0",
"9": "255.128.0.0",
"10":"255.192.0.0",
"11":"255.224.0.0",
"12":"255.240.0.0",
"13":"255.248.0.0",
"14":"255.252.0.0",
"15":"255.254.0.0",
"16":"255.255.0.0",
"17":"255.255.128.0",
"18":"255.255.192.0",
"19":"255.255.224.0",
"20":"255.255.240.0",
"21":"255.255.248.0",
"22":"255.255.252.0",
"23":"255.255.254.0",
"24":"255.255.255.0",
"25":"255.255.255.128",
"26":"255.255.255.192",
"27":"255.255.255.224",
"28":"255.255.255.240",
"29":"255.255.255.248",
"30":"255.255.255.252",
"31":"255.255.255.254",
"32":"255.255.255.255"
}

# Prefix to Number of hosts
P2H = {
"0": pow(2,32),
"1": pow(2,31),
"2": pow(2,30),
"3": pow(2,29),
"4": pow(2,28),
"5": pow(2,27),
"6": pow(2,26),
"7": pow(2,25),
"8": pow(2,24),
"9": pow(2,23),
"10": pow(2,22),
"11": pow(2,21),
"12": pow(2,20),
"13": pow(2,19),
"14": pow(2,18),
"15": pow(2,17),
"16": pow(2,16),
"17": pow(2,15),
"18": pow(2,14),
"19": pow(2,13),
"20": pow(2,12),
"21": pow(2,11),
"22": pow(2,10),
"23": pow(2,9),
"24": pow(2,8),
"25": pow(2,7),
"26": pow(2,6),
"27": pow(2,5),
"28": pow(2,4),
"29": pow(2,3),
"30": pow(2,2),
"31": pow(2,1),
"32": pow(2,0)
}

# Subnet to Host
S2H = {
"0.0.0.0": pow(2,32),
"128.0.0.0": pow(2,31),
"192.0.0.0": pow(2,30),
"224.0.0.0": pow(2,29),
"240.0.0.0": pow(2,28),
"248.0.0.0": pow(2,27),
"252.0.0.0": pow(2,26),
"254.0.0.0": pow(2,25),
"255.0.0.0": pow(2,24),
"255.128.0.0": pow(2,23),
"255.192.0.0": pow(2,22),
"255.224.0.0": pow(2,21),
"255.240.0.0": pow(2,20),
"255.248.0.0": pow(2,19),
"255.252.0.0": pow(2,18),
"255.254.0.0": pow(2,17),
"255.255.0.0": pow(2,16),
"255.255.128.0": pow(2,15),
"255.255.192.0": pow(2,14),
"255.255.224.0": pow(2,13),
"255.255.240.0": pow(2,12),
"255.255.248.0": pow(2,11),
"255.255.252.0": pow(2,10),
"255.255.254.0": pow(2,9),
"255.255.255.0": pow(2,8),
"255.255.255.128": pow(2,7),
"255.255.255.192": pow(2,6),
"255.255.255.224": pow(2,5),
"255.255.255.240": pow(2,4),
"255.255.255.248": pow(2,3),
"255.255.255.252": pow(2,2),
"255.255.255.254": pow(2,1),
"255.255.255.255": pow(2,0)
}

def getNB(prefix):
    """
    @param prefix: IP address prefix (ex. 192.168.0.0/24)
    @return (network, broadcast) address
    """
    i = prefix.split("/")
    ip = i[0]
    subnet = P2S[i[1]]
    n = network(ip, subnet)
    b = broadcast(ip, subnet)
    return (n, b)

def getNB2(prefix):
    (n,b) = getNB(prefix)
    return(IP2Int(n), IP2Int(b))

def IP2Int(ip):
    o = map(int, ip.split("."))
    return (16777216 * o[0]) + (65536 * o[1]) + (256 * o[2]) + o[3]

def Int2IP(ipnum):
    o1 = int(ipnum / 16777216) % 256
    o2 = int(ipnum / 65536) % 256
    o3 = int(ipnum / 256) % 256
    o4 = int(ipnum) % 256
    return "%(o1)s.%(o2)s.%(o3)s.%(o4)s" % locals()

def network(ip, subnet_mask):
    """
    @param ip: "a.b.c.d"
    @param subnet_mask: "xxx.xxx.xxx.xxx"
    @return: network address
    network = ip & subnet_mask
    """
    a = IP2Int(ip)
    b = IP2Int(subnet_mask)
    c = a & b
    return Int2IP(c)

def broadcast(ip, subnet_mask):
    a = IP2Int(ip)
    b = IP2Int(subnet_mask)
    s = a & b
    e = s + S2H[subnet_mask] - 1
    return Int2IP(e)

def overlap(p1, p2):
    """
    compare two prefix (p1, p2)
    if overlap p1 and p2
    return overlap IP address range
    """
    (a1,a2) = getNB2(p1)
    (b1,b2) = getNB2(p2)
    if a1 == b1:
        if a2 >= b2:
            return (True, (Int2IP(a1), Int2IP(b2)))
        else:
            return (True, (Int2IP(a1), Int2IP(a2)))
    elif a1 < b1:
        (c1,c2) = (a1,a2)
        (d1,d2) = (b1,b2)
    else:
        (c1,c2) = (b1,b2)
        (d1,d2) = (a1,a2)
    # Every time, c1 <= d1
    if c2 < d1:
        return (False, None)
    else:
        if c2 >= d2:
            return (True, (Int2IP(d1), Int2IP(d2)))
        return (True, (Int2IP(d1), Int2Ip(c2)))
            
if __name__ == "__main__":
    print network("192.168.1.1","255.255.0.0")
    print broadcast("192.168.1.2","255.255.0.0")
    p1 = "10.0.0.1/24"
    print getNB(p1)
