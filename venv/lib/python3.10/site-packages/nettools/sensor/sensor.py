#!/usr/bin/env python
import datetime
import logging
import socket
import signal
import sys
import threading
import time

import SocketServer

from optparse import OptionParser

DUMMY_LEN = 1024
DUMMY = "1" * DUMMY_LEN

KB = 1024
MB = KB * 1024
GB = MB * 1024

# Report Interval
INTV = 2.0

# Global Options
OPT = {'server':False, 'client':False, 'port':5001,
        'connect_to':'127.0.0.1'
    }

# Global variable
latency = 0
latency_history = []
num_threads = 0
c_stat = {}
STOPPED = True

def reporter(tid, duration, bw, last=False):
    # for client reporter
    global latency_history
    if OPT['server'] == True:
        pass
    # From here, client
    c_stat[tid] = [duration,bw]
    keys = c_stat.keys()
    keys.sort()
    if len(keys) == num_threads:
        # Every data is updated
        output = ""
        sum_bw = 0
        for key in keys:
            v = c_stat[key]
            bw = v[1]
            sum_bw = sum_bw + bw
            (f_bw, scale) = getScaled(bw)
            # [thread id]   [bandwidth] bps"
            output = output + " +[Thread:%s]\t%10.4f %sbps\n" % (key, f_bw, scale)
        # summary
        (f_bw, scale) = getScaled(sum_bw)
        dt = datetime.datetime.now()
        if last==True:
            avg_latency = 0
            for i in latency_history:
                avg_latency = avg_latency + i
            global latency
            latency = avg_latency / len(latency_history)
            clientFooter()
            summary = "%s seconds\t%10.4f %sbps\t%10.4f" % (OPT['time'], f_bw, scale, latency)
        else:
            summary = "%s\t%10.4f %sbps\t%10.4f" % (dt.strftime("%H:%M:%S"), f_bw, scale, latency)
        print summary
        print output
        # clear dic
        c_stat.clear()

        # logging to latency history
        latency_history.append(latency)

def clientTitle():
    print """
#################################################
Time                Bandwidth       Latency(ms)
-------------------------------------------------"""

def clientFooter():
    print """
------------ Summary ----------------------------"""

def getScaled(bw):
    # return scaled, bandwidth
    if bw > GB:
        f_bw = bw/GB
        scale = "G"
    elif bw > MB:
        f_bw = bw/MB
        scale = "M"
    elif bw > KB:
        f_bw = bw/KB
        scale = "K"
    else:
        scale = ""
    return (f_bw, scale)

def report(tid, s0, s2, total, last=False):
    # @param tid: Thread name
    # @param s0: start time (sec)
    # @param s2: end time (sec)
    # @param total: total sent (bytes)

    # TODO: Check s0 < s2
    duration = s2 - s0
    bandwidth = total * 8 / duration

    reporter(tid, duration, bandwidth, last)


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        data = self.request.recv(1024)
        logger.debug("[From Client]%s" % data)
        cur_thread = threading.current_thread()
        tid = cur_thread.ident
        response = "{}: {}".format(cur_thread.name, data)
        cmd = self.parseCmd(data)
        cmd0 = cmd[0]
        #################################
        # Download                      #
        # format: download|<duration>   #
        #################################
        if cmd0 == 'download':
            # Mark start time
            s0 = s2 = time.time()
            s1 = s0 + INTV
            s3 = s0 + int(cmd[1])
            # size calculation
            total = 0
            prev_total = 0
            while s2 <= s3:
                self.request.sendall(DUMMY)
                s2 = time.time()
                if s2 >= s1:
                    # Time to periodic report
                    report(tid, 0, INTV, total - prev_total)
                    s1 = s1 + INTV
                    prev_total = total
                total = total + DUMMY_LEN
            # End of Send (EOF)
            self.request.sendall("0")

            # Report Result
            report(tid, s0,s2,total)
        ####################################
        # Echo                             #
        # format: echo|<client_timestamp>  #
        ####################################
        elif cmd0 == 'echo':
            # appending server timestamp, then reply
            while True:
                reply = "%s|%f" % (data,time.time())
                logger.debug("ECHO reply=%s" % reply)
                self.request.sendall(reply)
                data = self.request.recv(128)
                if data[0] == "0":
                    break
            logger.info("End of ECHO Thread")
                
        else:
            logger.error("wrong request:%s" % cmd)

    def parseCmd(self, cmd):
        # Parse Client Command
        # Command format (delimiter=|)
        #  command|arg1|arg2|arg3|...
        # 
        # 1) download
        #  format = download|<duration seconds>
        #  ex) download|60
        #       download from server during 60 seconds
        #
        # 2) echo (for latency check)
        #  echo|sender_timesmap
        #
        # @Return : dictionary 
        idx = cmd.split("|")
        ret = {}
        for i in range(len(idx)):
            ret[i] = idx[i]
        return ret

       
class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

###########################################
# Client
###########################################
def client_exit(thrs):
    print "\n"
    logger.critical("******************************")
    logger.critical("Force stop threads")
    logger.critical("******************************")
    print "\n"
    global STOPPED
    STOPPED = False

    sys.exit()

class ClientManager:
    def __init__(self, args):
        # @param args (dictionary)
        #  - num_threads
        #  - ip
        #  - port
        #  - cmd
        self.threads = []
        self.num_threads = args['num_threads']
        ip = args['ip']
        port = args['port']
        cmd = args['cmd']

        # update global
        global num_threads
        num_threads = self.num_threads

        exit_handler = lambda signum, frame: client_exit(self.threads)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGQUIT, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        
        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)
        signal.signal(signal.SIGQUIT, exit_handler)
        signal.signal(signal.SIGHUP, exit_handler)


        # Start Download Threads
        for i in range(self.num_threads):
            self.startClient(i, ip, port, cmd)

        # Start 1 echo Thread
        t = threading.Thread(target=echoClient, args=(i+1, ip, port, cmd,))
        self.threads.append(t)
        t.start()

        # Wait Join
        #for t in self.threads:
        #    t.join()

    def startClient(self, tid, ip, port, cmd):
        t = threading.Thread(target=tcpClient, args=(tid, ip, port, cmd,))
        self.threads.append(t)
        t.start()

def tcpClient(tid, ip, port, cmd):
    logger.debug("TCP Client[%s] started" % tid)
    # cmd[0] = 'download'
    # cmd[1] = duration
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        # Send Command first
        sock.sendall("download|%s" % cmd[1])
        CONT = True
        s0 = time.time()
        s1 = s0 + INTV
        total = 0
        prev_total = 0
        while CONT and STOPPED:
            res = sock.recv(1024)
            s2 = time.time()
            if s2 >= s1:
                # Time to periodic report
                report(tid, 0, INTV, total - prev_total)
                s1 = s1 + INTV
                prev_total = total
            total = total + DUMMY_LEN
            if res[0] == "0":
                # End of Download
                CONT = False
        logger.debug("END of tcpClient Thread:%s" % tid)
    except:
        pass
    finally:
        s2 = time.time()
        report(tid, s0,s2,total,last=True)
        sock.close()

def echoClient(tid, ip, port, cmd):
    logger.debug("TCP Echo Client[%s] stared" % tid)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip,port))
    s0 = s1 = time.time()
    s3 = s0 + int(cmd[1])        # durtion
    global latency
    try:
        while s1 <= s3 and STOPPED:
            s1 = time.time()
            cmd = "echo|%s" % s1
            sock.send(cmd)
            recv = sock.recv(128)
            s2 = time.time()
            # Calculate latency (ms)
            # roudtrip time / 2
            latency = (s2 - s1) * 1000 / 2
            # Sleep interval
            time.sleep(INTV)
    finally:
        sock.send("0")
        sock.close()
 
def cleanup_and_exit(server):
    logger.debug("Clean Up Thread")
    server.shutdown()
    server.server_close()
    sys.exit()

def runAsServer():
    logger.info("Running Server")

    server = None
    exit_handler = lambda signum, frame: cleanup_and_exit(server)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGQUIT, exit_handler)
    signal.signal(signal.SIGHUP, exit_handler)

    server = ThreadedTCPServer((HOST, OPT['port']), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    logger.debug("Server loop running in thread:", server_thread.name)
    while True:
        # Parent Thread
        # Wait for signal
        time.sleep(5)
    
def runAsClient():
    logger.debug("Running Client")
    cmd = {0:'download',1:OPT['time']}

    cargs = {'num_threads':OPT['parallel'],
            'ip':OPT['connect_to'],
            'port':OPT['port'],
            'cmd':cmd}

    clientTitle()
    client = ClientManager(cargs)

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    parser = OptionParser()
    parser.add_option("-s","--server", dest="server", action="store_true", help="run in server mode")
    parser.add_option("-p","--port"  , dest="port"  , default = 5001, help="server port to listen on/connect to")

    parser.add_option("-c","--client", dest="client", metavar="host", help="run in client mode, connect to <host>")
    parser.add_option("-t","--time", dest="time", metavar="seconds", help="time in seconds to transmit for (default 10 secs)")
    parser.add_option("-P","--parallel",dest="parallel", metavar="num", help="number of parallel client threads to run")
    parser.add_option("-L","--logging",dest="logging", help="logging level(DEBUG|INFO|WARNING|ERROR|CRITICAL)")

    try:
        (options,args) = parser.parse_args()
        if options.logging:
            LEVEL = {}
            LEVEL['DEBUG'] = logging.DEBUG
            LEVEL['WARNING'] = logging.WARNING
            LEVEL['INFO'] = logging.INFO
            LEVEL['ERROR'] = logging.ERROR
            LEVEL['CRITICAL'] = logging.CRITICAL

            logging.basicConfig(format='%(levelname)s %(message)s', level=LEVEL[options.logging])
            logger = logging.getLogger('flywheel')
        else:
            logging.basicConfig(format='%(levelname)s %(message)s', level=logging.INFO)
            logger = logging.getLogger('flywheel')
        if options.server:
            OPT['server'] = True
        if options.client:
            OPT['client'] = True
            OPT['connect_to'] = options.client

        if options.port:
            OPT['port'] = int(options.port)

        if options.parallel:
            OPT['parallel'] = int(options.parallel)
        else:
            OPT['parallel'] = 1
      
        if options.time:
            OPT['time'] = int(options.time)
        else:
            OPT['time'] = 10
     
        HOST = ""

        if OPT['server']:
            runAsServer()

        if OPT['client']:
            runAsClient()

    except:
        #parser.print_help()
        sys.exit()
